from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from support_agent.capstone_config import RANDOM_SEED
from support_agent.training_data import normalize_prompt


def stratified_partitions(
    df: pd.DataFrame,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    if round(train_size + validation_size + test_size, 6) != 1:
        raise ValueError("train_size + validation_size + test_size must equal 1.0")

    _assert_enough_per_class(df, "intent", min_count=3)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["intent"],
    )
    relative_validation = validation_size / (validation_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_validation,
        random_state=seed,
        stratify=temp_df["intent"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def leakage_report(partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    names = list(partitions)
    normalized = {
        name: set(partitions[name]["prompt"].map(normalize_prompt))
        for name in names
    }
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = normalized[left].intersection(normalized[right])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_count": len(overlap),
                    "has_leakage": bool(overlap),
                }
            )
    return pd.DataFrame(rows)


def save_partitions(partitions: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in partitions.items():
        partition.to_json(output_dir / f"{name}.jsonl", orient="records", lines=True)


def _assert_enough_per_class(df: pd.DataFrame, column: str, min_count: int) -> None:
    counts = df[column].value_counts()
    too_small = counts[counts < min_count]
    if not too_small.empty:
        raise ValueError(
            "Each intent needs at least "
            f"{min_count} examples for stratified train/validation/test splits. "
            f"Too small: {too_small.to_dict()}"
        )
