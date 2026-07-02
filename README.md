# SHL Assessment Recommender

Conversational agent that turns a vague hiring need into a grounded shortlist of
SHL Individual Test Solutions. FastAPI service with `GET /health` and `POST /chat`,
per the assignment spec.

## Project layout

```
app/            FastAPI service, retrieval, agent orchestration, LLM client
data/catalog.json   normalized SHL catalog (built from catalog_raw.json)
scripts/build_catalog.py   regenerates data/catalog.json from the raw scrape
eval/           replays the 10 sample conversation traces, estimates Recall@10
tests/          unit tests (stub LLM, no network calls / API key needed)
sample_conversations/   the 10 SHL-provided dev traces, for reference
APPROACH.md     2-page design writeup
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env             # fill in LLM_API_KEY
```

Any OpenAI-compatible chat-completions provider works (see `.env.example`).
**Groq is recommended** for actual grading/deployment: its free tier allows far
more requests/minute than Gemini's free tier, which we found caps out at a
handful of requests per day on a fresh key -- workable for local smoke tests,
not for a real multi-turn eval run or live grading traffic.

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"We need a solution for senior leadership."}]}'
```

## Tests

```bash
python -m pytest tests/ -q
```

All 15 tests run against a stubbed LLM client (no network/API key needed) and
cover: catalog loading, BM25 + alias retrieval, and the agent's grounding/
schema-enforcement rules (refusal always empties recommendations, hallucinated
ids get filtered out, malformed JSON triggers one repair retry then a safe
fallback, etc).

## Eval against the sample conversations

```bash
python -m eval.run_eval          # needs LLM_API_KEY set
python -m eval.run_eval -v       # verbose: prints every turn
EVAL_CALL_DELAY_SECONDS=13 python -m eval.run_eval   # throttle for tight rate limits
```

This replays each of the 10 public traces' user turns (verbatim, in order) against
the live agent and reports an approximate Recall@10 per trace. It's an
approximation of the real evaluator: the real harness uses an LLM to *simulate*
a user who answers dynamically based on what our agent asks, whereas this
script just fires the fixed reference lines regardless of our agent's
questions. Good enough to catch retrieval/prompt regressions during
development; see APPROACH.md for how this was actually used.

## Rebuilding the catalog

```bash
python scripts/build_catalog.py   # reads catalog_raw.json -> writes data/catalog.json
```

## Deploy

- **Docker** (Render/Fly/Railway/anywhere): `Dockerfile` included, exposes port 8000.
- **Render**: `render.yaml` provided (Docker runtime, free plan, `/health` healthcheck).
  Set `LLM_API_KEY` in the Render dashboard (marked `sync: false` so it isn't committed).
- **Procfile**-based platforms (Railway/Heroku-style): `Procfile` included.

Cold start note: the evaluator allows up to 2 minutes for the first `/health` call
on a cold instance -- Render's free tier spins down after inactivity, so the first
request after idle will be slow; this is expected and accounted for in the spec.
