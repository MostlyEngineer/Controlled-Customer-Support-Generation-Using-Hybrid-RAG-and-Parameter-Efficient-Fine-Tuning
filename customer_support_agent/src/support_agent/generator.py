from __future__ import annotations

import re

from support_agent.schema import IntentRoute, RetrievedChunk


class GroundedResponseGenerator:
    def generate(self, message: str, route: IntentRoute, context: list[RetrievedChunk]) -> tuple[str, list[str], list[str]]:
        if not context:
            return (
                "I could not find a matching SOP for this request. I would log the case and route it to a supervisor.",
                ["Can you share the order, account, or case reference connected to this request?"],
                ["No retrieval context was available."],
            )

        source = context[0]
        bullets = _extract_action_lines(source.text)
        answer_lines = [
            _opening(route),
            "",
            "Based on the applicable SOP:",
        ]
        answer_lines.extend(f"- {line}" for line in bullets[:5])

        if route.needs_escalation:
            answer_lines.extend(
                [
                    "",
                    f"Escalation: this should be treated as {route.escalation_level} and handed off with identity status, actions taken, and the desired outcome.",
                ]
            )

        follow_ups = _follow_ups(route.category)
        warnings = _warnings(route.category)
        answer_lines.extend(
            [
                "",
                f"Source used: {source.title}.",
            ]
        )
        return "\n".join(answer_lines), follow_ups, warnings


def _opening(route: IntentRoute) -> str:
    openings = {
        "REFUND_POLICY": "I can help with the refund request and keep the timeline policy-grounded.",
        "SHIPPING_DELAYS": "I can help check whether the shipment is delayed and what the next step should be.",
        "ORDER_TRACKING": "I can help interpret the order status and tracking next step.",
        "PASSWORD_RESET": "I can guide the customer through a secure password reset.",
        "ACCOUNT_RECOVERY": "This needs identity verification before any account recovery action.",
        "BILLING_DISPUTES": "I can help investigate the charge before deciding whether a correction is needed.",
        "DATA_PRIVACY": "This request needs identity verification and careful privacy handling.",
        "TECHNICAL_TROUBLESHOOTING": "I can walk through first-line troubleshooting and identify whether escalation is needed.",
    }
    return openings.get(route.category, "I can help using the relevant support procedure.")


def _extract_action_lines(text: str) -> list[str]:
    numbered = _numbered_items(text)
    if numbered:
        return numbered

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip(" -\t")
        if not line or line.startswith("#"):
            continue
        if re.match(r"^\d+\.\s+", line):
            line = re.sub(r"^\d+\.\s+", "", line)
        if len(line) < 35 and ":" not in line:
            continue
        lines.append(line)
    if lines:
        return lines

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def _numbered_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current:
                items.append(" ".join(current))
                current = []
            continue
        if re.match(r"^\d+\.\s+", line):
            if current:
                items.append(" ".join(current))
            current = [re.sub(r"^\d+\.\s+", "", line).strip()]
            continue
        if current:
            current.append(line.strip(" -"))
    if current:
        items.append(" ".join(current))
    return [re.sub(r"\s+", " ", item).strip() for item in items if item]


def _follow_ups(category: str) -> list[str]:
    by_category = {
        "REFUND_POLICY": ["What is the order identifier and purchase or delivery date?", "Is the item unused, defective, or already consumed?"],
        "ORDER_TRACKING": ["What is the order identifier and registered email?", "What tracking status is currently shown?"],
        "SHIPPING_DELAYS": ["What was the estimated delivery window?", "When was the last carrier scan?"],
        "BILLING_DISPUTES": ["What is the charge date, amount, and descriptor?", "Has a bank chargeback already been opened?"],
        "ACCOUNT_RECOVERY": ["Can the customer verify two approved identity factors?", "Is there suspected unauthorized access?"],
        "TECHNICAL_TROUBLESHOOTING": ["What device, OS, browser, or app version is affected?", "Can the issue be reproduced?"],
    }
    return by_category.get(category, ["Can you share the relevant order, account, or case details?"])


def _warnings(category: str) -> list[str]:
    warnings = {
        "PAYMENT_METHODS": ["Never request full card numbers or security codes over chat or email."],
        "PASSWORD_RESET": ["Never ask for or set the customer's password."],
        "DATA_PRIVACY": ["Never disclose personal data before identity verification."],
        "BILLING_DISPUTES": ["Do not issue a manual refund if an active chargeback may cause double credit."],
    }
    return warnings.get(category, [])
