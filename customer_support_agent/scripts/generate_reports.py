from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


PROPOSAL = """
Project Scope

This project implements a controlled customer-support generation system using
three progressively enhanced systems: baseline intent generation, SOP-assisted
retrieval generation, and hybrid retrieval generation with a fine-tuned JSON
intent router.

The operational boundary is support guidance. The system classifies raw
customer queries, retrieves relevant corporate SOP passages, generates
grounded response guidance, and flags escalation needs. It does not directly
mutate accounts, process refunds, process payments, or override policies.

Expected router JSON:
{"intent": "refund_request", "category": "REFUND_POLICY"}

Methodology

Stage 1 profiles customer interaction data and SOP documents, including
prompt length, intent balance, duplicate detection, and SOP TF-IDF similarity.
Stage 2 standardizes records, creates strict ChatML targets, performs
normalized deduplication, and creates stratified train/validation/test splits.
Stage 3 establishes a deterministic baseline and a naive RAG system using SOP
retrieval. Stage 4 fine-tunes a LoRA/QLoRA router that emits strict JSON and
uses the extracted intent/category to improve retrieval.

Success Criteria

Router format adherence should be at least 95 percent. Intent accuracy should
improve after fine-tuning. SOP-grounded RAG should reduce unsupported policy
claims compared with baseline generation. Deterministic inference should
produce stable answers across repeated runs.
"""


COMPARATIVE = """
Comparative Analysis

Baseline

The baseline system measures how well a pre-trained instruction model can
classify customer-support intent and follow strict JSON formatting without
retrieved policy context. It is expected to be most vulnerable to formatting
failures and unsupported policy claims.

Naive RAG

The RAG system retrieves relevant SOP passages and uses them as grounding
context. This should reduce hallucinated refund windows, shipping timelines,
password handling mistakes, and escalation errors. Retrieval quality depends
on query wording and semantic similarity.

Hybrid RAG

The hybrid system uses a fine-tuned intent router to emit structured JSON
before retrieval. The category and intent can be appended to the search query
or used as a metadata filter, improving retrieval for messy, sarcastic, or
ambiguous customer messages.

Recommendations

Use one fixed model ID, embedding model, dataset split, and inference
configuration across all comparisons. Report format adherence, intent exact
match, ROUGE/BLEU, consistency, and hallucination frequency on the held-out
test split and adversarial subset only.
"""


def write_pdf(title: str, body: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        y = 0.95
        fig.text(0.08, y, title, fontsize=18, weight="bold", va="top")
        y -= 0.06
        for paragraph in body.strip().split("\n\n"):
            lines = []
            for line in paragraph.splitlines():
                lines.extend(wrap(line, width=88) or [""])
            block = "\n".join(lines)
            size = 12 if len(lines) == 1 and len(paragraph) < 40 else 9
            weight = "bold" if size == 12 else "normal"
            fig.text(0.08, y, block, fontsize=size, weight=weight, va="top", family="monospace")
            y -= 0.026 * max(1, len(lines)) + 0.018
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                y = 0.95
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    write_pdf("Project Proposal and Methodology", PROPOSAL, REPORTS / "Project_Proposal_and_Methodology.pdf")
    write_pdf("Comparative Analysis Report", COMPARATIVE, REPORTS / "Comparative_Analysis_Report.pdf")
    print("wrote", REPORTS / "Project_Proposal_and_Methodology.pdf")
    print("wrote", REPORTS / "Comparative_Analysis_Report.pdf")


if __name__ == "__main__":
    main()
