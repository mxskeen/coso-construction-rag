import os
import re
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.parser import ConstructionDocumentParser, DocumentChunk
from src.fine_tuning import load_construction_embedder


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r'[a-zA-Z0-9_\u0900-\u097F]+', text.lower())


class ConstructionHybridRetriever:
    def __init__(self, use_finetuned: bool = True, cache_dir: str = "data/index_cache"):
        self.use_finetuned = use_finetuned
        self.cache_dir = cache_dir
        self.embedder = load_construction_embedder(use_finetuned=use_finetuned)
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

    def build_index(self, pdf_paths: List[str], force_rebuild: bool = False):
        os.makedirs(self.cache_dir, exist_ok=True)
        model_tag = "adapted" if self.use_finetuned else "baseline"
        chunks_cache = os.path.join(self.cache_dir, "corpus_chunks.pkl")
        faiss_cache = os.path.join(self.cache_dir, f"faiss_{model_tag}.index")

        if not force_rebuild and os.path.exists(chunks_cache):
            with open(chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            parser = ConstructionDocumentParser()
            self.chunks = []
            for path in pdf_paths:
                self.chunks.extend(parser.parse_pdf(path))
            with open(chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)

        tokenized_corpus = [tokenize_for_bm25(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        if not force_rebuild and os.path.exists(faiss_cache):
            self.faiss_index = faiss.read_index(faiss_cache)
        else:
            texts = [c.content for c in self.chunks]
            embs = self.embedder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
            embs = np.array(embs, dtype=np.float32)
            self.faiss_index = faiss.IndexFlatIP(embs.shape[1])
            self.faiss_index.add(embs)
            faiss.write_index(self.faiss_index, faiss_cache)

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.faiss_index.search(np.array(q_emb, dtype=np.float32), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def retrieve(self, query: str, top_k: int = 5, hybrid: bool = True, rrf_k: int = 60) -> List[Dict[str, Any]]:
        if not self.chunks:
            raise ValueError("Index not built. Call build_index() first.")

        if not hybrid:
            dense_results = self.search_dense(query, top_k=top_k)
            return [{**self.chunks[idx].to_dict(), "score": score, "rank": rank}
                    for rank, (idx, score) in enumerate(dense_results, start=1)]

        candidate_k = max(top_k * 3, 20)
        dense_hits = self.search_dense(query, top_k=candidate_k)
        bm25_hits = self.search_bm25(query, top_k=candidate_k)

        rrf_scores: Dict[int, float] = {}
        dense_ranks: Dict[int, int] = {}
        bm25_ranks: Dict[int, int] = {}

        for rank, (idx, _) in enumerate(dense_hits, start=1):
            dense_ranks[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (idx, _) in enumerate(bm25_hits, start=1):
            bm25_ranks[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        return [{
            **self.chunks[idx].to_dict(),
            "rrf_score": round(rrf_scores[idx], 5),
            "dense_rank": dense_ranks.get(idx),
            "bm25_rank": bm25_ranks.get(idx),
            "rank": rank
        } for rank, idx in enumerate(sorted_indices, start=1)]
