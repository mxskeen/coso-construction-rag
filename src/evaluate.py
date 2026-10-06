import json
import os
from typing import List, Dict, Any
from src.hybrid_retriever import ConstructionHybridRetriever


def is_clause_match(retrieved_clause: str, target_clause: str) -> bool:
    rc = retrieved_clause.lower().replace(" ", "").replace(".", "")
    tc = target_clause.lower().replace(" ", "").replace(".", "")
    return tc in rc or rc in tc


def evaluate_retriever(retriever: ConstructionHybridRetriever, benchmarks: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    reciprocal_ranks = []

    for item in benchmarks:
        query = item["query"]
        target_doc = item["target_doc"]
        target_clause = item["target_clause"]
        results = retriever.retrieve(query, top_k=top_k)

        rank_found = 0
        for rank, res in enumerate(results, start=1):
            doc_match = target_doc.lower() in res["doc_name"].lower()
            clause_match = is_clause_match(res["clause_id"], target_clause) or is_clause_match(res["clause_title"], target_clause)
            if doc_match and clause_match:
                rank_found = rank
                break

        if rank_found == 1:
            hit_1 += 1
        if 1 <= rank_found <= 3:
            hit_3 += 1
        if 1 <= rank_found <= 5:
            hit_5 += 1

        reciprocal_ranks.append(1.0 / rank_found if rank_found > 0 else 0.0)

    n = len(benchmarks)
    return {
        "Hit@1": round((hit_1 / n) * 100, 2),
        "Hit@3": round((hit_3 / n) * 100, 2),
        "Hit@5": round((hit_5 / n) * 100, 2),
        "MRR@5": round(sum(reciprocal_ranks) / n, 4)
    }


def run_comparison(benchmarks_path: str = "data/eval_benchmarks.json"):
    with open(benchmarks_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    base = "data/corpus" if os.path.exists("data/corpus") else "RAG-Interview-Test-Corpus-20261005T065046Z-1-001/RAG-Interview-Test-Corpus"
    pdf_paths = [
        os.path.join(base, "IS456-2000_concrete-code_EN.pdf"),
        os.path.join(base, "CPWD-GCC-2020_construction-contract_EN.pdf"),
        os.path.join(base, "PMGSY-III-summary_HINDI_selectable-text.pdf")
    ]

    print(f"\nEvaluating on {len(benchmarks)} domain-specific construction questions...")

    print("\n--- [1] Evaluating BASELINE (all-MiniLM-L6-v2) ---")
    baseline_retriever = ConstructionHybridRetriever(use_finetuned=False)
    baseline_retriever.build_index(pdf_paths)
    base_metrics = evaluate_retriever(baseline_retriever, benchmarks)

    print("\n--- [2] Evaluating FINE-TUNED (Domain-Adapted Embedder) ---")
    adapted_retriever = ConstructionHybridRetriever(use_finetuned=True)
    adapted_retriever.build_index(pdf_paths)
    adapted_metrics = evaluate_retriever(adapted_retriever, benchmarks)

    print("\n" + "=" * 55)
    print("        RETRIEVAL EVALUATION RESULTS (BEFORE vs AFTER)")
    print("=" * 55)
    print(f"{'Metric':<12} | {'Baseline (Base)':<18} | {'Fine-Tuned (Adapted)':<20}")
    print("-" * 55)
    for m in ["Hit@1", "Hit@3", "Hit@5", "MRR@5"]:
        base_v = f"{base_metrics[m]}%" if "Hit" in m else f"{base_metrics[m]}"
        adapt_v = f"{adapted_metrics[m]}%" if "Hit" in m else f"{adapted_metrics[m]}"
        print(f"{m:<12} | {base_v:<18} | {adapt_v:<20}")
    print("=" * 55)


if __name__ == "__main__":
    run_comparison()
