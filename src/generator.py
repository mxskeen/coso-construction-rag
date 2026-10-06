import re
import json
import urllib.request
from typing import List, Dict, Any, Optional


class GroundedGenerator:
    def __init__(self, ollama_model: str = "qwen2.5-coder:7b-instruct-q4_K_M", ollama_url: str = "http://localhost:11434"):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

    def _call_ollama(self, prompt: str, timeout: int = 2) -> Optional[str]:
        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256}
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception:
            return None

    def _extractive_fallback(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant construction clauses found to answer this query."

        # If the top chunk is a Table, format the table content directly
        top = chunks[0]
        if "table" in top["clause_id"].lower() or "table" in top["clause_title"].lower() or "सारणी" in top["clause_id"].lower():
            lines = [l.strip() for l in top["content"].split("\n") if l.strip() and not l.startswith("[")]
            # Filter out non-table header noise
            content_preview = "\n".join(lines[:15])
            return f"Based on {top['full_citation']}, the relevant provision/table data states:\n\n{content_preview}"

        stopwords = {'what', 'is', 'the', 'for', 'under', 'to', 'in', 'and', 'a', 'an', 'of', 'how', 'much', 'can'}
        q_words = set(re.findall(r'[a-zA-Z0-9_\u0900-\u097F]+', query.lower())) - stopwords

        best_sentence, best_score, best_chunk = "", -1, chunks[0]

        for rank_idx, c in enumerate(chunks[:5]):
            # Boost score for top retrieved ranks: rank 0 gets +15, rank 1 gets +10, etc.
            rank_boost = (5 - rank_idx) * 3
            unwrapped = re.sub(r'(?<![\.\?\!।])\n', ' ', c["content"])
            sentences = [s.strip() for s in re.split(r'[\.\?\!।]\s+', unwrapped) if len(s.strip()) > 20 and not s.strip().startswith("[")]
            for s in sentences:
                s_words = set(re.findall(r'[a-zA-Z0-9_\u0900-\u097F]+', s.lower()))
                overlap = len(q_words & s_words)
                has_pct = 5 if re.search(r'(\d+\s*%|\bpercent\b|\bप्रतिशत\b|\d+:\d+)', s, re.I) else 0
                has_num = 2 if re.search(r'\d+', s) else 0
                score = rank_boost + (overlap * 4) + has_pct + has_num
                if score > best_score:
                    best_score, best_sentence, best_chunk = score, s, c

        if best_sentence:
            return f"Based on {best_chunk['full_citation']}, the relevant provision states:\n\"{best_sentence}.\""

        body = [l.strip() for l in top["content"].split("\n") if l.strip() and not l.startswith("[")]
        return f"Based on {top['full_citation']}, the relevant provision states:\n\"{' '.join(body[:3])}\""

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        citations = []
        context_blocks = []

        for i, c in enumerate(retrieved_chunks[:5], start=1):
            cite_str = f"{c['doc_name']} | {c['clause_id']} ({c['clause_title']}) | Page {c['page_number']}"
            citations.append({
                "source": c["doc_name"],
                "clause_id": c["clause_id"],
                "clause_title": c["clause_title"],
                "page": c["page_number"],
                "citation": c["full_citation"],
                "excerpt": c["content"][:250].replace("\n", " ").strip()
            })
            context_blocks.append(f"[{i}] {cite_str}\n{c['content']}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"You are a construction engineering assistant. Answer the user question strictly using "
            f"the provided context. Always cite the exact document, clause number, and page number.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        llm_answer = self._call_ollama(prompt, timeout=2)
        final_answer = llm_answer if llm_answer else self._extractive_fallback(query, retrieved_chunks)

        return {
            "query": query,
            "answer": final_answer,
            "citations": citations
        }
