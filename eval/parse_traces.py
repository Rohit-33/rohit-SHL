"""Parses the SHL-provided sample conversation traces (GenAI_SampleConversations/*.md)
into ordered user turns plus the final labeled shortlist (name + url), so run_eval.py
can replay them against our own agent and score Recall@10.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

TRACES_DIR = (
    Path(__file__).resolve().parent.parent
    / "sample_conversations"
    / "GenAI_SampleConversations"
)

USER_BLOCK_RE = re.compile(
    r"\*\*User\*\*\s*\n\n((?:>.*\n?)+)", re.MULTILINE
)
TABLE_ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|([^|]+)\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*<?(https?://\S+?)>?\s*\|",
    re.MULTILINE,
)


@dataclass
class Trace:
    trace_id: str
    user_turns: List[str] = field(default_factory=list)
    expected_shortlist: List[Tuple[str, str]] = field(default_factory=list)  # (name, url)


def _clean_quote_block(block: str) -> str:
    lines = [line.lstrip(">").strip() for line in block.strip().splitlines()]
    return " ".join(line for line in lines if line)


def parse_trace_file(path: Path) -> Trace:
    text = path.read_text(encoding="utf-8")

    user_turns = [_clean_quote_block(m.group(1)) for m in USER_BLOCK_RE.finditer(text)]

    # The last markdown table in the file is the final, confirmed shortlist.
    tables = re.findall(r"(\|[^\n]*\|\n(?:\|[^\n]*\|\n)+)", text)
    expected = []
    if tables:
        last_table = tables[-1]
        for row_match in TABLE_ROW_RE.finditer(last_table):
            name = row_match.group(1).strip()
            url = row_match.group(2).strip()
            expected.append((name, url))

    return Trace(trace_id=path.stem, user_turns=user_turns, expected_shortlist=expected)


def load_all_traces() -> List[Trace]:
    return [parse_trace_file(p) for p in sorted(TRACES_DIR.glob("C*.md"))]


if __name__ == "__main__":
    for trace in load_all_traces():
        print(f"=== {trace.trace_id} ===")
        print(f"{len(trace.user_turns)} user turns")
        for t in trace.user_turns:
            print(" -", t[:90])
        print(f"expected shortlist ({len(trace.expected_shortlist)}):")
        for name, url in trace.expected_shortlist:
            print("   *", name, "|", url)
        print()
