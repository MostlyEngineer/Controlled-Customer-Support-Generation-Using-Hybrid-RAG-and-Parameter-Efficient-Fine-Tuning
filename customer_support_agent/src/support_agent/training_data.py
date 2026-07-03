from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from support_agent.router import RULES

PROMPT_CANDIDATES = ("prompt", "instruction", "customer_query", "utterance", "text", "sentence")
INTENT_CANDIDATES = ("intent", "intent_name", "label", "category_intent")
CATEGORY_CANDIDATES = ("category", "label_category", "department")


def load_instruction_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        return synthetic_instruction_dataset()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def standardize_instruction_schema(df: pd.DataFrame) -> pd.DataFrame:
    prompt_col = _find_column(df, PROMPT_CANDIDATES)
    intent_col = _find_column(df, INTENT_CANDIDATES)
    category_col = _find_column(df, CATEGORY_CANDIDATES)

    if prompt_col is None:
        raise ValueError(f"Could not find prompt column. Tried: {PROMPT_CANDIDATES}")

    out = pd.DataFrame()
    out["prompt"] = df[prompt_col].astype(str).str.strip()
    out["intent"] = df[intent_col].astype(str).str.strip().str.lower() if intent_col else "unknown_intent"
    out["category"] = (
        df[category_col].astype(str).str.strip().str.upper()
        if category_col
        else out["intent"].str.upper()
    )
    out["target_json"] = out.apply(
        lambda row: json.dumps({"intent": row["intent"], "category": row["category"]}, sort_keys=True),
        axis=1,
    )
    return out


def clean_instruction_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = ["prompt", "intent", "category", "target_json"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required standardized columns: {missing}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=required)
    cleaned["prompt"] = cleaned["prompt"].astype(str).str.strip()
    cleaned["intent"] = cleaned["intent"].astype(str).str.strip().str.lower()
    cleaned["category"] = cleaned["category"].astype(str).str.strip().str.upper()
    cleaned = cleaned[cleaned["prompt"].str.len() > 0]
    cleaned["normalized_prompt"] = cleaned["prompt"].map(normalize_prompt)
    cleaned = cleaned.drop_duplicates(subset=["normalized_prompt"]).reset_index(drop=True)
    cleaned["target_json"] = cleaned.apply(
        lambda row: json.dumps({"intent": row["intent"], "category": row["category"]}, sort_keys=True),
        axis=1,
    )
    return cleaned


def to_chatml(df: pd.DataFrame, system_prompt: str) -> pd.DataFrame:
    chatml = df.copy()
    chatml["messages"] = chatml.apply(
        lambda row: [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["target_json"]},
        ],
        axis=1,
    )
    chatml["chatml_text"] = chatml["messages"].map(render_chatml)
    return chatml


def render_chatml(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" for msg in messages)


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.casefold()).strip()


def synthetic_instruction_dataset() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    templates = [
        "I need help with {topic}.",
        "Can you explain the policy for {topic}?",
        "This is frustrating, I still need support for {topic}.",
        "Please route my case about {topic}.",
        "Nobody has solved my {topic} issue yet.",
        "I never got a clear answer about {topic}.",
        "The customer is asking about {topic}.",
        "What should an agent do for {topic}?",
        "I have a support ticket related to {topic}.",
        "Please classify this request: {topic}.",
    ]
    for rule in RULES:
        topic = rule.category.replace("_", " ").lower()
        for template in templates:
            rows.append(
                {
                    "prompt": template.format(topic=topic),
                    "intent": rule.intent,
                    "category": rule.category,
                }
            )
    df = pd.DataFrame(rows)
    return standardize_instruction_schema(df)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None
