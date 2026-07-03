# Customer Support Hybrid RAG Agent

This codebase implements a grounded customer-support agent for the capstone SOP corpus.
It includes:

- Intent routing to strict JSON.
- Retrieval over Markdown SOP documents.
- SOP-grounded response generation.
- CLI and FastAPI entry points.
- Evaluation helpers for format adherence, intent accuracy, consistency, and hallucination checks.
- Capstone notebooks and PDF report starters matching the required deliverables.

The project runs offline with scikit-learn TF-IDF retrieval. If you later add an instruction-tuning CSV
or a fine-tuned router, plug it in behind the same `SupportAgent` interface.

## Setup

```powershell
cd "C:\Users\sysadmin\Downloads\045e707b-f0fa-4904-9b52-48b1c04c5f1b-Capstone-Starter-Package\Workflow + Dataset\customer_support_agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For the full GPU fine-tuning and Chroma workflow, use:

```powershell
python -m pip install -r requirements-capstone.txt
```

## Run The Agent

```powershell
python -m support_agent.cli ask "My order says delivered but I never received it"
```

Interactive mode:

```powershell
python -m support_agent.cli chat
```

API server:

```powershell
uvicorn support_agent.api:app --reload
```

Then POST to `http://127.0.0.1:8000/ask`:

```json
{
  "message": "I was charged twice and need help"
}
```

## Dataset Layout

By default the code reads SOP files from:

```text
../Dataset/Dataset/sop_documents
```

Override it with:

```powershell
$env:SOP_DIR="C:\path\to\sop_documents"
```

## Capstone Mapping

- Stage 2 retrieval corpus preparation: `support_agent.documents`
- Stage 3 naive RAG: `support_agent.retrieval`, `support_agent.generator`
- Stage 4 intent router integration: `support_agent.router`
- Evaluation framework: `support_agent.evaluation`

Generated deliverables:

```text
notebooks/Project_Proposal_and_Methodology.ipynb
notebooks/Data_Understanding_and_EDA.ipynb
notebooks/Data_Preparation.ipynb
notebooks/Baseline_Model_Evaluation.ipynb
notebooks/RAG_Implementation.ipynb
notebooks/Solution_V1_RAG_Evaluation.ipynb
notebooks/Fine_Tuning_Pipeline.ipynb
notebooks/Solution_V2_FineTuned_RAG_Evaluation.ipynb
notebooks/Comparative_Analysis_Report.ipynb
reports/Project_Proposal_and_Methodology.pdf
reports/Comparative_Analysis_Report.pdf
```

Regenerate notebooks and PDFs:

```powershell
python scripts\generate_notebooks.py
python scripts\generate_reports.py
```

## Adding The Bitext Dataset

The attached files only include SOP Markdown documents, so the notebooks use a small deterministic
synthetic instruction dataset as a runnable placeholder. For final capstone scoring, download or place
the Bitext CSV locally and replace:

```python
raw_df = load_instruction_dataset(None)
```

with:

```python
raw_df = load_instruction_dataset(r"C:\path\to\bitext_customer_support.csv")
```

The loader will try to infer common prompt, intent, and category columns, then standardize records into:

```text
prompt, intent, category, target_json
```

The included router is deterministic and rule-based so the project is runnable from the supplied files.
For a true LoRA/QLoRA capstone submission, train a model to emit the same JSON schema:

```json
{"intent": "refund_request", "category": "REFUND_POLICY"}
```

Then replace `RuleBasedIntentRouter.route()` with model inference.
