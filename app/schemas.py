"""Request/response models for the /chat contract.

The shape here is intentionally exactly what the assignment specifies --
the evaluator's harness depends on it verbatim, so nothing is added or
renamed even where a richer shape would be nicer (e.g. duration, keys).
"""
from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: str = "ok"
