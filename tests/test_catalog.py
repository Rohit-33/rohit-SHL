from app.catalog import Catalog


def test_catalog_loads():
    c = Catalog()
    assert len(c) > 300


def test_lookup_by_id():
    c = Catalog()
    item = c.items[0]
    assert c.get(item["id"])["name"] == item["name"]


def test_get_many_skips_missing():
    c = Catalog()
    valid_id = c.items[0]["id"]
    result = c.get_many([valid_id, "does-not-exist"])
    assert len(result) == 1
