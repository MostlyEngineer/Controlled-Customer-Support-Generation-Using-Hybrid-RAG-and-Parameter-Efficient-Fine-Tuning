from __future__ import annotations

from pydantic import BaseModel, Field


class IntentRoute(BaseModel):
    intent: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_escalation: bool = False
    escalation_level: str = "P4"
    rationale: str = ""


class RetrievedChunk(BaseModel):
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    score: float
    text: str


class AgentResponse(BaseModel):
    route: IntentRoute
    answer: str
    retrieved_context: list[RetrievedChunk]
    follow_up_questions: list[str] = []
    warnings: list[str] = []
