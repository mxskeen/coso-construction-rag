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
        top = chunks[0]
        lines = [l.strip() for l in top["content"].split("\n") if l.strip()]
        body_lines = [l for l in lines if not l.startswith("[") and len(l) > 20]
        highlight = " ".join(body_lines[:3]) if body_lines else top["content"][:300]
        return f"Based on {top['full_citation']}, the relevant provision states:\n\"{highlight}\""

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        citations = []
        context_blocks = []

        for i, c in enumerate(retrieved_chunks[:3], start=1):
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
