from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from support_agent.agent import SupportAgent
from support_agent.schema import AgentResponse


app = FastAPI(title="Customer Support Hybrid RAG Agent", version="0.1.0")
agent = SupportAgent()


class AskRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AgentResponse)
def ask(request: AskRequest) -> AgentResponse:
    return agent.ask(request.message)
