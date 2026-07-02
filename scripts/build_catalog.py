"""
Normalizes the raw SHL catalog scrape (catalog_raw.json) into data/catalog.json,
the flat, LLM- and retrieval-friendly shape the app actually consumes.

Run with: python scripts/build_catalog.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "catalog_raw.json"
OUT_PATH = ROOT / "data" / "catalog.json"

# SHL's "Great 8" category -> single-letter test type code, as shown on
# shl.com product pages and in the assignment's own example rows.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}
CODE_ORDER = "ABCDEKPS"


def parse_duration_minutes(duration_raw: str):
    if not duration_raw:
        return None
    m = re.search(r"(\d+)", duration_raw)
    return int(m.group(1)) if m else None


def display_duration(duration_raw: str, minutes):
    if not duration_raw:
        return ""
    if minutes is not None:
        return f"{minutes} minutes"
    if "variable" in duration_raw.lower():
        return "Variable"
    if "untimed" in duration_raw.lower():
        return "Untimed"
    return duration_raw


def build():
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"), strict=False)

    items = []
    seen_ids = set()
    for entry in raw:
        entity_id = str(entry.get("entity_id") or "").strip()
        name = (entry.get("name") or "").strip()
        url = (entry.get("link") or "").strip()
        if not entity_id or not name or not url:
            continue
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)

        categories = entry.get("keys") or []
        codes = sorted(
            {CATEGORY_TO_CODE[c] for c in categories if c in CATEGORY_TO_CODE},
            key=lambda c: CODE_ORDER.index(c),
        )

        duration_raw = (entry.get("duration_raw") or entry.get("duration") or "").strip()
        duration_minutes = parse_duration_minutes(duration_raw)

        items.append(
            {
                "id": entity_id,
                "name": name,
                "url": url,
                "description": (entry.get("description") or "").strip(),
                "categories": categories,
                "test_type": ",".join(codes),
                "job_levels": entry.get("job_levels") or [],
                "languages": entry.get("languages") or [],
                "duration_minutes": duration_minutes,
                "duration_display": display_duration(duration_raw, duration_minutes),
                "remote_testing": (entry.get("remote") or "").strip().lower() == "yes",
                "adaptive_irt": (entry.get("adaptive") or "").strip().lower() == "yes",
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} catalog items to {OUT_PATH}")


if __name__ == "__main__":
    build()
