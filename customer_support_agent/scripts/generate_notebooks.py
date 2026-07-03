from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(filename: str, cells: list[nbf.NotebookNode]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NOTEBOOKS / filename)


SETUP = code(
    """
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from support_agent.capstone_config import *
    from support_agent.config import AgentConfig

    config = AgentConfig()
    print("Project root:", PROJECT_ROOT)
    print("SOP directory:", config.sop_dir)
    print("Model:", MODEL_ID)
    print("Embedding model:", EMBEDDING_MODEL_ID)
    """
)


def proposal() -> list[nbf.NotebookNode]:
    return [
        md("# Project Proposal and Methodology\n\nUse this notebook as the source for the proposal/report PDF."),
        SETUP,
        md(
            """
            ## Scope

            Target use case: customer-support intent routing and SOP-grounded response generation.

            Inputs:
            - Raw customer support query.
            - SOP Markdown corpus.

            Router output:

            ```json
            {"intent": "refund_request", "category": "REFUND_POLICY"}
            ```

            Boundaries:
            - The system does not process payments, mutate accounts, or approve exceptions directly.
            - It provides grounded guidance and escalation recommendations based on SOP text.
            """
        ),
        md(
            """
            ## Success Criteria

            - Router JSON format adherence >= 95%.
            - Intent exact-match accuracy improves from baseline to fine-tuned router.
            - RAG output reduces hallucination frequency compared with baseline generation.
            - Final responses cite or derive from retrieved SOP context.
            """
        ),
        code(
            """
            from support_agent.fine_tuning import LoRAConfig, fine_tuning_steps

            print(LoRAConfig())
            for step in fine_tuning_steps():
                print("-", step)
            """
        ),
    ]


def eda() -> list[nbf.NotebookNode]:
    return [
        md("# Data Understanding and EDA"),
        SETUP,
        code(
            """
            from support_agent.training_data import load_instruction_dataset, standardize_instruction_schema, clean_instruction_dataset
            from support_agent.eda import dataset_profile, sop_similarity_matrix

            raw_df = load_instruction_dataset(None)  # Replace None with the Bitext CSV path when available.
            df = clean_instruction_dataset(standardize_instruction_schema(raw_df))
            profile = dataset_profile(df)
            profile
            """
        ),
        code(
            """
            import pandas as pd
            pd.Series(profile["intent_distribution"]).sort_values(ascending=False).plot(kind="bar", title="Intent Distribution")
            """
        ),
        code(
            """
            import seaborn as sns
            import matplotlib.pyplot as plt

            sim = sop_similarity_matrix(config.sop_dir)
            plt.figure(figsize=(10, 8))
            sns.heatmap(sim, cmap="viridis")
            plt.title("SOP TF-IDF Cosine Similarity")
            plt.show()
            sim.round(3)
            """
        ),
    ]


def prep() -> list[nbf.NotebookNode]:
    return [
        md("# Data Preparation"),
        SETUP,
        code(
            """
            from support_agent.training_data import load_instruction_dataset, standardize_instruction_schema, clean_instruction_dataset, to_chatml
            from support_agent.data_prep import stratified_partitions, leakage_report, save_partitions

            raw_df = load_instruction_dataset(None)  # Replace with Bitext CSV path.
            df = clean_instruction_dataset(standardize_instruction_schema(raw_df))
            chatml_df = to_chatml(df, SYSTEM_PROMPT)
            partitions = stratified_partitions(chatml_df)
            leakage = leakage_report(partitions)
            leakage
            """
        ),
        code(
            """
            output_dir = ARTIFACTS_DIR / "partitions"
            save_partitions(partitions, output_dir)
            {name: len(partition) for name, partition in partitions.items()}
            """
        ),
        code(
            """
            # Tokenization template for the full capstone environment.
            # Install requirements-capstone.txt, then uncomment.
            #
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # token_counts = chatml_df["chatml_text"].map(lambda x: len(tokenizer(x).input_ids))
            # token_counts.describe()
            """
        ),
    ]


def baseline() -> list[nbf.NotebookNode]:
    return [
        md("# Baseline Model Evaluation"),
        SETUP,
        code(
            """
            from support_agent.baseline import baseline_prompt, parse_router_json
            from support_agent.router import RuleBasedIntentRouter

            prompts = [
                "I want a refund for a defective item",
                "My order is late and tracking has not moved",
                "I forgot my password",
            ]

            # Offline deterministic baseline used for local smoke testing.
            router = RuleBasedIntentRouter()
            rows = []
            for prompt in prompts:
                route = router.route(prompt)
                rows.append({"prompt": prompt, "intent": route.intent, "category": route.category})
            rows
            """
        ),
        code(
            """
            # Full LLM baseline in GPU/Colab:
            # from support_agent.baseline import run_transformer_baseline
            # output = run_transformer_baseline(prompts[0], MODEL_ID)
            # print(output)
            # print(parse_router_json(output))
            """
        ),
    ]


def rag_impl() -> list[nbf.NotebookNode]:
    return [
        md("# RAG Implementation"),
        SETUP,
        code(
            """
            from support_agent.documents import chunk_documents
            from support_agent.retrieval import TfidfRetriever

            chunks = chunk_documents(config.sop_dir)
            retriever = TfidfRetriever(chunks)
            results = retriever.search("My package was delivered but I did not receive it", top_k=3)
            [(r.title, round(r.score, 3)) for r in results]
            """
        ),
        code(
            """
            # Optional Chroma + all-MiniLM-L6-v2 index:
            # from support_agent.vector_store import build_chroma_index
            # count = build_chroma_index(config.sop_dir, ARTIFACTS_DIR / "chroma")
            # print("Indexed chunks:", count)
            """
        ),
    ]


def rag_eval() -> list[nbf.NotebookNode]:
    return [
        md("# Solution V1 RAG Evaluation"),
        SETUP,
        code(
            """
            from support_agent.agent import SupportAgent
            from support_agent.evaluation import consistency_rate, hallucination_frequency

            agent = SupportAgent()
            prompts = [
                "I want a refund for a defective item",
                "My order says delivered but I never received it",
                "I was charged twice",
            ]
            responses = [agent.ask(prompt) for prompt in prompts]
            answers = [response.answer for response in responses]
            {
                "consistency_rate": consistency_rate(agent, prompts),
                "hallucination_frequency": hallucination_frequency(answers),
            }
            """
        ),
        code("responses[0].model_dump()"),
    ]


def fine_tuning() -> list[nbf.NotebookNode]:
    return [
        md("# Fine-Tuning Pipeline"),
        SETUP,
        code(
            """
            from support_agent.fine_tuning import LoRAConfig, peft_training_import_check, fine_tuning_steps

            print(LoRAConfig())
            print(peft_training_import_check())
            for step in fine_tuning_steps():
                print("-", step)
            """
        ),
        code(
            """
            # Full PEFT implementation belongs in a GPU runtime.
            # This notebook documents the exact configuration boundary and expected training loop:
            # - load train/validation JSONL from artifacts/partitions
            # - tokenize ChatML
            # - mask labels before assistant answer with -100
            # - attach LoRA adapters
            # - train with AdamW and early stopping on validation loss
            # - save best adapter checkpoint and loss curves
            """
        ),
    ]


def hybrid_eval() -> list[nbf.NotebookNode]:
    return [
        md("# Solution V2 Fine-Tuned RAG Evaluation"),
        SETUP,
        code(
            """
            from support_agent.agent import SupportAgent
            from support_agent.evaluation import format_adherence_rate, intent_accuracy, hallucination_frequency

            # Replace SupportAgent.router with a fine-tuned router adapter once trained.
            agent = SupportAgent()
            prompts = ["I still have not received my refund", "Terrible, my order never arrived"]
            responses = [agent.ask(prompt) for prompt in prompts]
            router_json = [response.route.model_dump_json() for response in responses]
            {
                "format_adherence_rate": format_adherence_rate(router_json),
                "hallucination_frequency": hallucination_frequency([r.answer for r in responses]),
                "routes": [r.route.model_dump() for r in responses],
            }
            """
        ),
    ]


def comparative() -> list[nbf.NotebookNode]:
    return [
        md("# Comparative Analysis Report"),
        SETUP,
        code(
            """
            import pandas as pd

            comparison = pd.DataFrame([
                {"system": "Baseline", "format_adherence": None, "intent_accuracy": None, "hallucination_frequency": None},
                {"system": "Naive RAG", "format_adherence": "N/A", "intent_accuracy": "N/A", "hallucination_frequency": "measure in V1"},
                {"system": "Hybrid RAG", "format_adherence": "measure in V2", "intent_accuracy": "measure in V2", "hallucination_frequency": "measure in V2"},
            ])
            comparison
            """
        ),
        md(
            """
            ## Findings Template

            - Baseline establishes how often the pre-trained model follows JSON and policy constraints.
            - Naive RAG measures the independent impact of SOP retrieval on factual grounding.
            - Hybrid RAG measures the additional impact of fine-tuned intent routing on retrieval quality.

            ## Limitations

            Replace fallback synthetic examples with the Bitext instruction-tuning dataset for final scoring.
            """
        ),
    ]


def main() -> None:
    notebooks = {
        "Project_Proposal_and_Methodology.ipynb": proposal(),
        "Data_Understanding_and_EDA.ipynb": eda(),
        "Data_Preparation.ipynb": prep(),
        "Baseline_Model_Evaluation.ipynb": baseline(),
        "RAG_Implementation.ipynb": rag_impl(),
        "Solution_V1_RAG_Evaluation.ipynb": rag_eval(),
        "Fine_Tuning_Pipeline.ipynb": fine_tuning(),
        "Solution_V2_FineTuned_RAG_Evaluation.ipynb": hybrid_eval(),
        "Comparative_Analysis_Report.ipynb": comparative(),
    }
    for filename, cells in notebooks.items():
        write_notebook(filename, cells)
        print("wrote", filename)


if __name__ == "__main__":
    main()
