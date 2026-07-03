from support_agent.agent import SupportAgent


def test_refund_route_and_answer_are_grounded():
    agent = SupportAgent()
    response = agent.ask("I want a refund for my order from last week")

    assert response.route.intent == "refund_request"
    assert response.route.category == "REFUND_POLICY"
    assert response.retrieved_context
    assert "Source used" in response.answer


def test_billing_escalation_for_fraud():
    agent = SupportAgent()
    response = agent.ask("There is payment fraud and an unauthorized charge on my card")

    assert response.route.needs_escalation is True
    assert response.route.escalation_level == "P1"


def test_agent_is_deterministic():
    agent = SupportAgent()
    first = agent.ask("My package is stuck with no movement").answer
    second = agent.ask("My package is stuck with no movement").answer

    assert first == second
