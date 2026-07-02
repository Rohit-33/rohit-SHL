from app.catalog import Catalog
from app.retrieval import RetrievalIndex


def test_search_finds_java():
    index = RetrievalIndex(Catalog())
    results = index.search("senior java developer spring rest api", top_k=10)
    names = [r["name"].lower() for r in results]
    assert any("java" in n for n in names)


def test_alias_match_opq():
    index = RetrievalIndex(Catalog())
    matched = index.alias_matches("we should use OPQ for this")
    assert any("opq32r" in m["name"].lower() for m in matched)


def test_candidates_combines_search_and_alias():
    index = RetrievalIndex(Catalog())
    results = index.candidates("hiring a java developer, also compare to GSA", top_k=10)
    names = [r["name"].lower() for r in results]
    assert any("java" in n for n in names)
    assert any("global skills assessment" in n for n in names)
