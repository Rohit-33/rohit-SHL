"""Core orchestration: retrieve candidates, prompt the LLM, validate and
ground its decision, and produce a response that satisfies the /chat
contract no matter what the LLM does.
"""
import json
import logging
import re
from typing import List, Optional

from app.catalog import Catalog, get_catalog
from app.llm_client import LLMClient, LLMError
from app.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from app.retrieval import RetrievalIndex, get_index
from app.schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 25
MAX_RECOMMENDATIONS = 10

FALLBACK_RESPONSE = ChatResponse(
    reply=(
        "Sorry, I hit an internal snag processing that. Could you rephrase what kind of "
        "role or assessment need you're looking at?"
    ),
    recommendations=[],
    end_of_conversation=False,
)

SCOPE_REFUSAL_TEXT = (
    "I can only help with selecting SHL assessments -- I'm not able to help with that. "
    "Tell me about the role or skills you're hiring for and I can suggest a shortlist."
)


def _transcript(messages: List[Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _retrieval_query(messages: List[Message]) -> str:
    user_texts = [m.content for m in messages if m.role == "user"]
    all_texts = [m.content for m in messages]
    last_user = user_texts[-1] if user_texts else ""
    # Recency-weighted: last user turn counts most, then all user turns,
    # then the full transcript (assistant turns carry constraint context too,
    # e.g. "For selection with a leadership benchmark...").
    parts = [last_user] * 3 + user_texts + all_texts
    return "\n".join(parts)


CANDIDATE_LEGEND = (
    "Test type codes: A=Ability&Aptitude, B=Biodata/SituationalJudgment, C=Competencies, "
    "D=Development&360, E=AssessmentExercises, K=Knowledge&Skills, P=Personality&Behavior, "
    "S=Simulations."
)


def _format_candidates(items: List[dict]) -> str:
    lines = []
    for item in items:
        job_levels = ", ".join(item.get("job_levels", [])[:3])
        desc = (item.get("description") or "")[:100].replace("\n", " ")
        adaptive = " | adaptive" if item.get("adaptive_irt") else ""
        duration = item.get("duration_display") or "-"
        lines.append(
            f'{item["id"]} | {item["name"]} | type={item["test_type"] or "-"} | '
            f"levels={job_levels or '-'} | duration={duration}{adaptive} | {desc}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in LLM output")


def _build_llm_messages(messages: List[Message], candidates: List[dict]) -> List[dict]:
    user_content = (
        f"Conversation so far:\n{_transcript(messages)}\n\n"
        f"{CANDIDATE_LEGEND}\n"
        f"CANDIDATE ASSESSMENTS (id | name | type | levels | duration | desc):\n"
        f"{_format_candidates(candidates)}\n\n"
        "Respond with the JSON object now."
    )
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_messages.extend(FEW_SHOT_EXAMPLES)
    llm_messages.append({"role": "user", "content": user_content})
    return llm_messages


def _ground_recommendations(
    recommended_ids: List[str], candidates: List[dict], catalog: Catalog
) -> List[Recommendation]:
    candidate_ids = {c["id"] for c in candidates}
    out = []
    for rid in recommended_ids:
        rid = str(rid)
        if rid not in candidate_ids:
            continue
        item = catalog.get(rid)
        if not item:
            continue
        out.append(Recommendation(name=item["name"], url=item["url"], test_type=item["test_type"] or "-"))
        if len(out) >= MAX_RECOMMENDATIONS:
            break
    return out


class RecommenderAgent:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        catalog: Optional[Catalog] = None,
        index: Optional[RetrievalIndex] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.catalog = catalog or get_catalog()
        self.index = index or get_index()

    def _call_llm_once(self, llm_messages: List[dict]) -> dict:
        raw = self.llm_client.complete_json(llm_messages)
        return _extract_json(raw)

    def respond(self, messages: List[Message]) -> ChatResponse:
        query_text = _retrieval_query(messages)
        candidates = self.index.candidates(query_text, top_k=MAX_CANDIDATES)

        if not candidates:
            # Nothing matched at all yet (e.g. very first, very vague turn) --
            # still give the LLM a representative slice so it isn't grounding
            # against nothing, but keep it small.
            candidates = self.catalog.items[:MAX_CANDIDATES]

        llm_messages = _build_llm_messages(messages, candidates)

        parsed = None
        last_error = None
        for attempt in range(2):
            try:
                parsed = self._call_llm_once(llm_messages)
                break
            except (ValueError, json.JSONDecodeError) as e:
                # Bad output shape -- worth one repair round-trip within the same call.
                last_error = e
                logger.warning("LLM parse attempt %d failed: %s", attempt, e)
                llm_messages = llm_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON matching the required "
                            "schema. Respond again with ONLY the JSON object, no other text."
                        ),
                    }
                ]
            except LLMError as e:
                # Transport/provider failure (timeout, 429, 5xx): retrying immediately
                # won't help within the request's time budget, so fail straight to
                # the safe fallback instead of burning the remaining budget.
                last_error = e
                logger.warning("LLM transport failure: %s", e)
                break

        if parsed is None:
            logger.error("LLM failed after retries: %s", last_error)
            return FALLBACK_RESPONSE

        reply = str(parsed.get("reply") or "").strip()
        in_scope = bool(parsed.get("in_scope", True))
        commit_shortlist = bool(parsed.get("commit_shortlist", False))
        end_of_conversation = bool(parsed.get("end_of_conversation", False))
        recommended_ids = parsed.get("recommended_ids") or []
        if not isinstance(recommended_ids, list):
            recommended_ids = []

        if not in_scope:
            commit_shortlist = False
            end_of_conversation = False
            if not reply:
                reply = SCOPE_REFUSAL_TEXT

        if not commit_shortlist:
            recommendations: List[Recommendation] = []
            end_of_conversation = False
        else:
            recommendations = _ground_recommendations(recommended_ids, candidates, self.catalog)
            if not recommendations:
                # Model claimed a shortlist but nothing grounded -- treat as
                # not-yet-committed rather than returning an empty "success".
                commit_shortlist = False
                end_of_conversation = False

        if not reply:
            reply = "Could you tell me a bit more about the role you're hiring for?"

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )
