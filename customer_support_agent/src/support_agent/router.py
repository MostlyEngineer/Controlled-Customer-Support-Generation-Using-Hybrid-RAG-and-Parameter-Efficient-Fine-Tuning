from __future__ import annotations

import re
from dataclasses import dataclass

from support_agent.schema import IntentRoute


@dataclass(frozen=True)
class IntentRule:
    intent: str
    category: str
    keywords: tuple[str, ...]


RULES: tuple[IntentRule, ...] = (
    IntentRule("account_recovery", "ACCOUNT_RECOVERY", ("locked", "lockout", "recover account", "hacked", "unauthorized", "lost email")),
    IntentRule("billing_dispute", "BILLING_DISPUTES", ("charged", "charge", "duplicate", "billing", "invoice", "dispute", "pre-authorization")),
    IntentRule("data_privacy_request", "DATA_PRIVACY", ("delete my data", "personal data", "privacy", "gdpr", "access copy", "marketing opt")),
    IntentRule("order_tracking", "ORDER_TRACKING", ("where is my order", "tracking", "delivered", "parcel", "package", "carrier")),
    IntentRule("password_reset", "PASSWORD_RESET", ("password", "reset", "forgot password", "sign in", "login")),
    IntentRule("payment_method_help", "PAYMENT_METHODS", ("payment method", "card", "declined", "wallet", "bank transfer", "cash on delivery")),
    IntentRule("product_return", "PRODUCT_RETURN", ("return", "exchange", "wrong item", "defective", "return label")),
    IntentRule("refund_request", "REFUND_POLICY", ("refund", "money back", "reimburse", "partial refund", "store credit")),
    IntentRule("shipping_delay", "SHIPPING_DELAYS", ("late", "delayed", "stuck", "no movement", "lost shipment", "shipping delay")),
    IntentRule("subscription_cancellation", "SUBSCRIPTION_CANCELLATION", ("cancel subscription", "auto-renew", "pause subscription", "recurring", "renewal")),
    IntentRule("technical_troubleshooting", "TECHNICAL_TROUBLESHOOTING", ("crash", "error", "bug", "sync", "checkout error", "slow", "not working")),
    IntentRule("working_hours", "WORKING_HOURS", ("hours", "open", "callback", "weekend", "holiday", "phone support", "live chat")),
)

P1_TERMS = ("fraud", "account takeover", "data breach", "unauthorized access", "major outage", "hacked")
P2_TERMS = ("legal", "regulator", "lawsuit", "vulnerable", "data loss")
P3_TERMS = ("policy exception", "high value", "reproducible bug", "supervisor")


class RuleBasedIntentRouter:
    def route(self, message: str) -> IntentRoute:
        normalized = _normalize(message)
        best_rule = max(RULES, key=lambda rule: _score_rule(normalized, rule))
        best_score = _score_rule(normalized, best_rule)

        if best_score == 0:
            return IntentRoute(
                intent="general_support",
                category="ESCALATION_MATRIX",
                confidence=0.35,
                needs_escalation=False,
                escalation_level="P4",
                rationale="No exact support intent matched; defaulting to general SOP lookup.",
            )

        escalation_level = _escalation_level(normalized)
        needs_escalation = escalation_level in {"P1", "P2", "P3"}
        confidence = min(0.95, 0.45 + (best_score * 0.12))
        return IntentRoute(
            intent=best_rule.intent,
            category=best_rule.category,
            confidence=round(confidence, 2),
            needs_escalation=needs_escalation,
            escalation_level=escalation_level,
            rationale=f"Matched {best_score} keyword signal(s) for {best_rule.category}.",
        )


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", message.casefold()).strip()


def _score_rule(normalized: str, rule: IntentRule) -> int:
    score = 0
    for keyword in rule.keywords:
        if keyword in normalized:
            score += 2 if " " in keyword else 1
    return score


def _escalation_level(normalized: str) -> str:
    if any(term in normalized for term in P1_TERMS):
        return "P1"
    if any(term in normalized for term in P2_TERMS):
        return "P2"
    if any(term in normalized for term in P3_TERMS):
        return "P3"
    return "P4"
