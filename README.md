# Construction RAG with Retrieval Domain Adaptation

A focused RAG pipeline built over Indian construction documents (engineering standards, statutory contracts, and bilingual scheme policies). Features **clause-aware chunking**, **hybrid retrieval (BM25 + FAISS)**, **exact clause citations**, and a **contrastively fine-tuned embedding model** with empirical before-and-after evaluation.

---

## Architecture

<img width="1447" height="823" alt="image" src="https://github.com/user-attachments/assets/98cab406-5a0a-4459-80ee-29000d64129f" />


1. **Parser**: Extracts text while stripping headers/noise, preserves clause hierarchies (`Clause 2`, `Table 16`), and prepends document/clause breadcrumbs to each chunk.
2. **Fine-Tuning**: Adapts a sentence-transformer (`all-MiniLM-L6-v2`) on construction triplets using `MultipleNegativesRankingLoss` (InfoNCE).
3. **Hybrid Retrieval**: Combines FAISS dense semantic search and BM25 lexical search using Reciprocal Rank Fusion (RRF).
4. **Generator**: Synthesizes grounded answers with exact document, clause, page citations, and evidence excerpts.

---

## Documents Used

| Document | Type | Key Characteristics |
| :--- | :--- | :--- |
| **`IS456-2000_concrete-code_EN.pdf`** | Technical Standard | Plain & reinforced concrete specifications, minimum grades, cover rules, curing periods. |
| **`CPWD-GCC-2020_construction-contract_EN.pdf`** | Construction Contract | Statutory conditions, delay penalties (`Clause 2`), defect liability (`Clause 17`), arbitration (`Clause 25`). |
| **`PMGSY-III-summary_HINDI_selectable-text.pdf`** | Bilingual Policy | Rural road scheme guidelines in Hindi, fund-sharing ratios (60:40, 90:10), bridge limits. |

---

## Setup & Quickstart

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ask a Question
Query via CLI:
```bash
# Contract delay penalty question
python query.py "What is the penalty for delay under CPWD contracts?"

# Technical engineering standard question
python query.py "What is the minimum nominal cover for severe exposure condition?"
```

Or enter interactive mode:
```bash
python query.py
```

> **Generation & LLM Note**:
> By default, the system runs with **zero external dependencies** — if no local LLM daemon is running, `src/generator.py` uses a table- and rank-aware extractive fallback that surfaces verbatim operative clauses with citations. If you have [Ollama](https://ollama.ai) installed (`ollama run qwen2.5-coder`), the generator automatically detects `http://localhost:11434` and uses it for grounded synthesis.

### 3. Run Benchmark Evaluation
To reproduce the retrieval evaluation comparing the base model vs. the fine-tuned model:
```bash
python run_eval.py
```

### 4. Run Automated Tests
```bash
python -m unittest tests/test_pipeline.py
```

---

## Evaluation Results (Before vs. After Fine-Tuning)

Evaluated on 15 domain-specific construction questions across all three documents:

| Metric | Baseline (`all-MiniLM-L6-v2`) | Fine-Tuned (Domain-Adapted) | Gain |
| :--- | :--- | :--- | :--- |
| **Hit@1** | 40.00% | **53.33%** | **+13.33%** |
| **Hit@3** | 73.33% | **86.67%** | **+13.34%** |
| **Hit@5** | 86.67% | **93.33%** | **+6.66%** |
| **MRR@5** | 0.5744 | **0.6911** | **+20.3% relative gain** |

---

## Sample Output

```text
[Question]: What is the penalty for delay under CPWD contracts?
------------------------------------------------------------
[ANSWER]:
Based on CPWD-GCC-2020_construction-contract_EN.pdf, Page 16-17, Clause 2 (General Conditions):
The contractor is liable to pay compensation for delay calculated on a per-day basis, subject to a maximum of 10% of the tendered contract value.

============================================================
CITATIONS & EVIDENCE:
============================================================

[1] CPWD-GCC-2020_construction-contract_EN.pdf, Page 17, Clause 2 (General Conditions)
    RRF Score: 0.03279 (Dense Rank: 1, BM25 Rank: 1)
    Evidence Excerpt: "[CPWD-GCC-2020_construction-contract_EN.pdf | Clause 2: General Conditions | Page 17] Provided that compensation during the progress of work before the justified extended date..."
------------------------------------------------------------
```

---

## Project Structure

```
├── src/
│   ├── parser.py           # Layout & clause-aware document segmentation
│   ├── dataset_builder.py  # Construction triplet data & eval benchmarks
│   ├── fine_tuning.py      # Contrastive domain adaptation training loop
│   ├── hybrid_retriever.py # BM25 + FAISS retriever with Reciprocal Rank Fusion
│   ├── generator.py        # Grounded answer synthesis and citation formatting
│   └── evaluate.py         # Retrieval benchmark metrics (Hit@k, MRR@k)
├── data/
│   ├── corpus/             # Selected PDF documents
│   └── eval_benchmarks.json# Ground-truth evaluation questions
├── assets/                 # Architecture diagram
├── tests/                  # Unit test suite
├── query.py                # User CLI interface
├── run_eval.py             # Benchmark runner
├── requirements.txt        # Pinned dependencies
└── README.md
```
