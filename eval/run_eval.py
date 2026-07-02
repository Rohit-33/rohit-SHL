"""Replays the 10 public sample conversations against our own live agent and
reports an approximate Recall@10, plus turn-by-turn transcripts for manual
inspection.

Approximation, by design: the real evaluator uses an LLM to *simulate* a user
who answers our agent's questions dynamically. Here we just replay the
sample trace's own user lines verbatim, in order, regardless of what our
agent asks -- good enough to sanity-check retrieval/prompt quality during
development without burning extra LLM calls on a simulated user, but it will
understate performance on traces where our agent's question order diverges
from the reference conversation's.

Usage: python -m eval.run_eval        (needs LLM_API_KEY set)
"""
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from app.agent import FALLBACK_RESPONSE, RecommenderAgent
from app.schemas import Message
from eval.parse_traces import Trace, load_all_traces

MAX_TURNS = 8  # user+assistant combined, matching the evaluator's cap
# Free-tier LLM rate limits (e.g. Gemini's 5 requests/minute) are far tighter than
# what a full eval run needs -- set EVAL_CALL_DELAY_SECONDS to throttle our own
# calls and avoid 429s drowning out the actual quality signal.
CALL_DELAY_SECONDS = float(os.environ.get("EVAL_CALL_DELAY_SECONDS", "0"))


def recall_at_10(expected: list, predicted_urls: set) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for _, url in expected if url in predicted_urls)
    return hits / len(expected)


def run_trace(agent: RecommenderAgent, trace: Trace, verbose: bool = False):
    messages: list[Message] = []
    last_recommendation_urls: set = set()
    turns_used = 0
    had_llm_failure = False

    for user_text in trace.user_turns:
        if turns_used >= MAX_TURNS:
            break
        messages.append(Message(role="user", content=user_text))
        turns_used += 1

        if CALL_DELAY_SECONDS:
            time.sleep(CALL_DELAY_SECONDS)
        resp = agent.respond(messages)
        if resp.reply == FALLBACK_RESPONSE.reply:
            had_llm_failure = True
        messages.append(Message(role="assistant", content=resp.reply))
        turns_used += 1

        if resp.recommendations:
            last_recommendation_urls = {r.url for r in resp.recommendations}

        if verbose:
            print(f"  USER: {user_text}")
            print(f"  AGENT: {resp.reply}")
            if resp.recommendations:
                for r in resp.recommendations:
                    print(f"    - {r.name} ({r.test_type}) {r.url}")
            print(f"  end_of_conversation={resp.end_of_conversation}\n")

        if resp.end_of_conversation:
            break

    return last_recommendation_urls, had_llm_failure


def main():
    verbose = "-v" in sys.argv
    agent = RecommenderAgent()
    traces = load_all_traces()

    scores = []
    inconclusive = 0
    for trace in traces:
        print(f"=== {trace.trace_id} ===")
        predicted_urls, had_failure = run_trace(agent, trace, verbose=verbose)
        if had_failure:
            inconclusive += 1
            print("  SKIPPED (LLM transport failure mid-trace, e.g. rate limit -- not a quality signal)\n")
            continue
        score = recall_at_10(trace.expected_shortlist, predicted_urls)
        scores.append(score)
        expected_urls = {url for _, url in trace.expected_shortlist}
        print(f"expected: {len(expected_urls)} | predicted: {len(predicted_urls)} | recall@10: {score:.2f}")
        missed = expected_urls - predicted_urls
        if missed:
            print("  missed:", missed)
        print()

    if scores:
        print(f"Mean Recall@10 across {len(scores)} scored traces: {sum(scores) / len(scores):.3f}")
    if inconclusive:
        print(f"{inconclusive} trace(s) skipped as inconclusive (LLM transport failure).")


if __name__ == "__main__":
    main()
