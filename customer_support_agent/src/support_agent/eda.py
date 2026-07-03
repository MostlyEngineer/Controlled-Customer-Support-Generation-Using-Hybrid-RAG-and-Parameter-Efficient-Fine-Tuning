from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from support_agent.documents import load_markdown_documents


def dataset_profile(df: pd.DataFrame) -> dict[str, object]:
    prompt_lengths = df["prompt"].astype(str).str.split().map(len)
    return {
        "records": int(len(df)),
        "intents": int(df["intent"].nunique()),
        "categories": int(df["category"].nunique()),
        "duplicate_prompts": int(df["prompt"].duplicated().sum()),
        "mean_prompt_words": float(prompt_lengths.mean()),
        "p95_prompt_words": float(prompt_lengths.quantile(0.95)),
        "intent_distribution": df["intent"].value_counts().to_dict(),
    }


def sop_similarity_matrix(sop_dir: Path) -> pd.DataFrame:
    docs = load_markdown_documents(sop_dir)
    names = [path.stem for path, _, _ in docs]
    texts = [text for _, _, text in docs]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
    similarities = cosine_similarity(matrix)
    return pd.DataFrame(similarities, index=names, columns=names)
