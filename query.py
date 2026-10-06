#!/usr/bin/env python3
import sys
import os
from src.hybrid_retriever import ConstructionHybridRetriever
from src.generator import GroundedGenerator


def _get_corpus_paths():
    base = "data/corpus" if os.path.exists("data/corpus") else "RAG-Interview-Test-Corpus-20261005T065046Z-1-001/RAG-Interview-Test-Corpus"
    return [
        os.path.join(base, "IS456-2000_concrete-code_EN.pdf"),
        os.path.join(base, "CPWD-GCC-2020_construction-contract_EN.pdf"),
        os.path.join(base, "PMGSY-III-summary_HINDI_selectable-text.pdf")
    ]

DEFAULT_PDFS = _get_corpus_paths()


def ask_question(query: str, retriever: ConstructionHybridRetriever, generator: GroundedGenerator):
    print(f"\n[Question]: {query}\n" + "-" * 60)
    
    # 1. Retrieve top-k chunks
    chunks = retriever.retrieve(query, top_k=3, hybrid=True)
    if not chunks:
        print("No matching clauses found in corpus.")
        return

    # 2. Generate grounded answer
    result = generator.generate(query, chunks)

    print("[ANSWER]:")
    print(result["answer"])
    print("\n" + "=" * 60)
    print("CITATIONS & EVIDENCE:")
    print("=" * 60)

    for i, cite in enumerate(result["citations"], start=1):
        chunk_meta = chunks[i - 1]
        print(f"\n[{i}] {cite['citation']}")
        print(f"    RRF Score: {chunk_meta.get('rrf_score', 'N/A')} (Dense Rank: {chunk_meta.get('dense_rank')}, BM25 Rank: {chunk_meta.get('bm25_rank')})")
        print(f"    Evidence Excerpt: \"{cite['excerpt']}...\"")
    print("-" * 60 + "\n")


def main():
    retriever = ConstructionHybridRetriever(use_finetuned=True)
    retriever.build_index(DEFAULT_PDFS)
    generator = GroundedGenerator()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        ask_question(query, retriever, generator)
    else:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nEnter construction query: ").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit", "q"]:
                    break
                ask_question(query, retriever, generator)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
