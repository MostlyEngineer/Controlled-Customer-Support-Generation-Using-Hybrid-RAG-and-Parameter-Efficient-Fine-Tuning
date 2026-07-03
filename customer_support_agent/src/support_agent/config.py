from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    sop_dir: Path = Field(default_factory=lambda: _default_sop_dir())
    top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "3")))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "900")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))


def _default_sop_dir() -> Path:
    env_path = os.getenv("SOP_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    return (project_root.parent / "Dataset" / "Dataset" / "sop_documents").resolve()
