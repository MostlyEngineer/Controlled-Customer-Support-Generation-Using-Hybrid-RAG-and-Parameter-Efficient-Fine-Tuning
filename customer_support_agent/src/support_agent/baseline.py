from __future__ import annotations

import json


def baseline_prompt(message: str) -> str:
    return (
        "Classify this customer-support query. Return only JSON with keys "
        f"intent and category.\n\nCustomer query: {message}"
    )


def run_transformer_baseline(message: str, model_id: str) -> str:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise ImportError(
            "Install optional capstone dependencies first: "
            "python -m pip install -r requirements-capstone.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    inputs = tokenizer(baseline_prompt(message), return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True).strip()


def parse_router_json(text: str) -> dict[str, str] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "intent" not in parsed or "category" not in parsed:
        return None
    return {"intent": str(parsed["intent"]), "category": str(parsed["category"])}
