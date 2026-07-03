from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    patience: int = 2


def peft_training_import_check() -> dict[str, bool]:
    modules = ["torch", "transformers", "datasets", "peft", "accelerate"]
    availability: dict[str, bool] = {}
    for module in modules:
        try:
            __import__(module)
            availability[module] = True
        except ImportError:
            availability[module] = False
    return availability


def fine_tuning_steps() -> list[str]:
    return [
        "Load train/validation JSONL partitions created in Data_Preparation.ipynb.",
        "Load tokenizer for the configured MODEL_ID and set a padding token if needed.",
        "Render ChatML messages and mask labels before assistant output with -100.",
        "Load the base causal LM with device_map='auto' and optional 4-bit quantization in Colab/Linux.",
        "Attach LoRA adapters with the configured target modules.",
        "Train with AdamW, evaluate validation loss each epoch, and save the best checkpoint.",
        "Plot train vs validation loss and record configuration plus random seed.",
    ]
