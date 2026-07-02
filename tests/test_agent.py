import json

from app.agent import RecommenderAgent
from app.catalog import Catalog
from app.retrieval import RetrievalIndex
from app.schemas import Message


class StubLLMClient:
    """Returns pre-scripted JSON responses in order, bypassing the network."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, messages, temperature=0.2, max_tokens=1500):
        self.calls.append(messages)
        return self.responses.pop(0)


def _agent(responses):
    catalog = Catalog()
    index = RetrievalIndex(catalog)
    return RecommenderAgent(llm_client=StubLLMClient(responses), catalog=catalog, index=index)


def test_clarify_turn_has_no_recommendations():
    agent = _agent(
        [
            json.dumps(
                {
                    "reply": "Who is this for -- what seniority level?",
                    "in_scope": True,
                    "commit_shortlist": False,
                    "recommended_ids": [],
                    "end_of_conversation": False,
                }
            )
        ]
    )
    resp = agent.respond([Message(role="user", content="We need a solution for senior leadership.")])
    assert resp.recommendations == []
    assert resp.end_of_conversation is False
    assert "?" in resp.reply


def test_commit_shortlist_grounds_ids_to_catalog():
    catalog = Catalog()
    real_item = catalog.items[0]
    agent = _agent(
        [
            json.dumps(
                {
                    "reply": "Here is a shortlist.",
                    "in_scope": True,
                    "commit_shortlist": True,
                    "recommended_ids": [real_item["id"], "fake-id-not-in-catalog"],
                    "end_of_conversation": True,
                }
            )
        ]
    )
    agent.catalog = catalog
    resp = agent.respond([Message(role="user", content="Recommend something.")])
    assert len(resp.recommendations) == 1
    assert resp.recommendations[0].name == real_item["name"]
    assert resp.recommendations[0].url == real_item["url"]
    assert resp.end_of_conversation is True


def test_refusal_forces_empty_and_not_ended():
    agent = _agent(
        [
            json.dumps(
                {
                    "reply": "That's a legal question outside my scope.",
                    "in_scope": False,
                    "commit_shortlist": True,  # model misbehaving on purpose
                    "recommended_ids": ["1", "2"],
                    "end_of_conversation": True,  # model misbehaving on purpose
                }
            )
        ]
    )
    resp = agent.respond([Message(role="user", content="Are we legally required to test staff?")])
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_hallucinated_ids_only_downgrades_to_no_commit():
    agent = _agent(
        [
            json.dumps(
                {
                    "reply": "Here is a shortlist.",
                    "in_scope": True,
                    "commit_shortlist": True,
                    "recommended_ids": ["totally-made-up"],
                    "end_of_conversation": True,
                }
            )
        ]
    )
    resp = agent.respond([Message(role="user", content="Recommend something.")])
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_malformed_json_retries_then_succeeds():
    agent = _agent(
        [
            "not json at all",
            json.dumps(
                {
                    "reply": "Ok, here goes.",
                    "in_scope": True,
                    "commit_shortlist": False,
                    "recommended_ids": [],
                    "end_of_conversation": False,
                }
            ),
        ]
    )
    resp = agent.respond([Message(role="user", content="hello")])
    assert resp.reply == "Ok, here goes."


def test_llm_fails_completely_returns_safe_fallback():
    agent = _agent(["still not json", "still not json either"])
    resp = agent.respond([Message(role="user", content="hello")])
    assert resp.recommendations == []
    assert resp.end_of_conversation is False
    assert resp.reply
