import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.agent import FALLBACK_RESPONSE, RecommenderAgent
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: RecommenderAgent | None = None


def get_agent() -> RecommenderAgent:
    global _agent
    if _agent is None:
        _agent = RecommenderAgent()
    return _agent


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        agent = get_agent()
        return agent.respond(request.messages)
    except Exception:
        logger.exception("Unhandled error in /chat")
        return FALLBACK_RESPONSE
