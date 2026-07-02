"""Loads the normalized SHL catalog and exposes lookup helpers.

This is the single source of truth for assessment data. Every URL and name
returned by the API is read from here at request time -- the LLM only ever
picks catalog IDs, never free-text names/URLs -- so the agent cannot
hallucinate a product that isn't in the catalog.
"""
import json
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


class Catalog:
    def __init__(self, path: Path = DATA_PATH):
        self.items = json.loads(path.read_text(encoding="utf-8"))
        self.by_id = {item["id"]: item for item in self.items}
        self.by_name_lower = {item["name"].lower(): item for item in self.items}

    def get(self, item_id: str) -> Optional[dict]:
        return self.by_id.get(str(item_id))

    def get_many(self, item_ids) -> list:
        out = []
        for iid in item_ids:
            item = self.get(iid)
            if item:
                out.append(item)
        return out

    def __len__(self):
        return len(self.items)


_catalog: Optional[Catalog] = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog
