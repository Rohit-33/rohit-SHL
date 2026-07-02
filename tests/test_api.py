import json

import app.main as main_module
from app.agent import RecommenderAgent
from app.catalog import Catalog
from app.retrieval import RetrievalIndex
from fastapi.testclient import TestClient
from tests.test_agent import StubLLMClient


def _client_with_stub(responses):
    catalog = Catalog()
    index = RetrievalIndex(catalog)
    stub_agent = RecommenderAgent(llm_client=StubLLMClient(responses), catalog=catalog, index=index)
    main_module._agent = stub_agent
    return TestClient(main_module.app)


def test_health():
    client = _client_with_stub([])
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_schema_shape():
    client = _client_with_stub(
        [
            json.dumps(
                {
                    "reply": "Who is this for?",
                    "in_scope": True,
                    "commit_shortlist": False,
                    "recommended_ids": [],
                    "end_of_conversation": False,
                }
            )
        ]
    )
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(body["recommendations"], list)
    assert isinstance(body["end_of_conversation"], bool)


def test_chat_rejects_bad_role():
    client = _client_with_stub([])
    resp = client.post("/chat", json={"messages": [{"role": "system", "content": "hi"}]})
    assert resp.status_code == 422
