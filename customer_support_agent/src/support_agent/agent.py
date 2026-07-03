from __future__ import annotations

from support_agent.config import AgentConfig
from support_agent.documents import chunk_documents
from support_agent.generator import GroundedResponseGenerator
from support_agent.retrieval import TfidfRetriever
from support_agent.router import RuleBasedIntentRouter
from support_agent.schema import AgentResponse


class SupportAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        chunks = chunk_documents(
            self.config.sop_dir,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.router = RuleBasedIntentRouter()
        self.retriever = TfidfRetriever(chunks)
        self.generator = GroundedResponseGenerator()

    def ask(self, message: str) -> AgentResponse:
        route = self.router.route(message)
        context = self.retriever.search(message, top_k=self.config.top_k, category_hint=route.category)
        answer, follow_ups, warnings = self.generator.generate(message, route, context)
        return AgentResponse(
            route=route,
            answer=answer,
            retrieved_context=context,
            follow_up_questions=follow_ups,
            warnings=warnings,
        )
