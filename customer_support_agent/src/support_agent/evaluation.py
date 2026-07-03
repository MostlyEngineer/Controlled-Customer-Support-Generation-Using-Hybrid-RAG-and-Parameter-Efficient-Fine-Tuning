from __future__ import annotations

import json
import re
from collections.abc import Iterable

from support_agent.agent import SupportAgent


def format_adherence_rate(json_strings: Iterable[str]) -> float:
    values = list(json_strings)
    if not values:
        return 0.0
    valid = 0
    for value in values:
        try:
            parsed = json.loads(value)
            valid += int(isinstance(parsed, dict) and "intent" in parsed and "category" in parsed)
        except json.JSONDecodeError:
            pass
    return valid / len(values)


def intent_accuracy(predicted: Iterable[str], expected: Iterable[str]) -> float:
    pairs = list(zip(predicted, expected, strict=False))
    if not pairs:
        return 0.0
    correct = sum(1 for pred, exp in pairs if pred == exp)
    return correct / len(pairs)


def consistency_rate(agent: SupportAgent, prompts: Iterable[str], repeats: int = 3) -> float:
    prompts = list(prompts)
    if not prompts:
        return 0.0
    stable = 0
    for prompt in prompts:
        outputs = [agent.ask(prompt).answer for _ in range(repeats)]
        stable += int(len(set(outputs)) == 1)
    return stable / len(prompts)


def hallucination_flags(answer: str) -> list[str]:
    flags: list[str] = []
    lower = answer.casefold()
    unsupported_patterns = [
        r"\bguarantee(?:d)? delivery\b",
        r"\brefund will post today\b",
        r"\bwe accept cash on delivery\b",
        r"\bshare your full card number\b",
        r"\bsend me your password\b",
    ]
    for pattern in unsupported_patterns:
        if re.search(pattern, lower):
            flags.append(pattern)
    return flags


def hallucination_frequency(answers: Iterable[str]) -> float:
    answers = list(answers)
    if not answers:
        return 0.0
    flagged = sum(1 for answer in answers if hallucination_flags(answer))
    return flagged / len(answers)
