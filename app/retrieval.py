"""Lexical retrieval over the SHL catalog.

377 items is small enough that a full embeddings/vector-DB stack is
overkill for a service that must cold-start in under a Render free-tier
minute and answer in well under the 30s per-call budget. BM25 over a
plain-text index gives fast, dependency-light, good-enough recall, and it
is fully deterministic and inspectable -- helpful when defending design
choices. See APPROACH.md for the tradeoff discussion.
"""
import re
from typing import List

from rank_bm25 import BM25Okapi

from app.catalog import Catalog, get_catalog

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Common abbreviations/nicknames that show up in conversation but not
# verbatim in catalog names -- used to force-include the right item as a
# retrieval candidate even when BM25's bag-of-words score is weak (short
# acronyms match poorly on term frequency alone).
ALIASES = {
    "opq": "occupational personality questionnaire opq32r",
    "opq32": "occupational personality questionnaire opq32r",
    "opq32r": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "dsi": "dependability and safety instrument",
    "mq": "motivation questionnaire",
    "g+": "verify interactive g+",
    "verify g+": "shl verify interactive g+",
    "svar": "svar spoken",
}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class RetrievalIndex:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._docs_tokens = []
        for item in catalog.items:
            text = " ".join(
                [
                    item["name"],
                    item["name"],
                    item["name"],  # boost name matches
                    item.get("description", ""),
                    " ".join(item.get("categories", [])),
                    " ".join(item.get("job_levels", [])),
                ]
            )
            self._docs_tokens.append(tokenize(text))
        self.bm25 = BM25Okapi(self._docs_tokens)

    def search(self, query: str, top_k: int = 40) -> List[dict]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            results.append(self.catalog.items[i])
        return results

    def alias_matches(self, text: str) -> List[dict]:
        text_lower = (text or "").lower()
        matched = []
        seen_ids = set()

        for alias, target_substr in ALIASES.items():
            if alias in text_lower:
                for item in self.catalog.items:
                    if target_substr in item["name"].lower() and item["id"] not in seen_ids:
                        matched.append(item)
                        seen_ids.add(item["id"])

        # Direct substring match: any catalog item whose full name (or the
        # name with trailing "(New)"/parenthetical stripped) appears in the
        # conversation verbatim -- catches exact product names typed by the
        # user, e.g. "Core Java (Advanced Level)".
        for item in self.catalog.items:
            if item["id"] in seen_ids:
                continue
            base_name = re.sub(r"\s*\([^)]*\)\s*", " ", item["name"]).strip().lower()
            if len(base_name) >= 6 and base_name in text_lower:
                matched.append(item)
                seen_ids.add(item["id"])

        return matched

    def candidates(self, query_text: str, top_k: int = 40) -> List[dict]:
        by_score = self.search(query_text, top_k=top_k)
        forced = self.alias_matches(query_text)

        seen_ids = {item["id"] for item in by_score}
        combined = list(by_score)
        for item in forced:
            if item["id"] not in seen_ids:
                combined.append(item)
                seen_ids.add(item["id"])
        return combined


_index: "RetrievalIndex | None" = None


def get_index() -> RetrievalIndex:
    global _index
    if _index is None:
        _index = RetrievalIndex(get_catalog())
    return _index
