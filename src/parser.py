import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import pymupdf


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_name: str
    doc_category: str
    page_number: int
    clause_id: str
    clause_title: str
    content: str
    full_citation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConstructionDocumentParser:
    def __init__(self, max_chunk_chars: int = 1000, chunk_overlap_chars: int = 150):
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def clean_text(self, text: str) -> str:
        text = re.sub(r'1234567890[0-9\s]*', '', text)
        text = re.sub(r'165\s+Years\s+of\s+Engineering\s+Excellence', '', text, flags=re.IGNORECASE)
        text = re.sub(r'bathfu;jh\s+mRd`’Vrk\s+ds\s+165\s+o’kZ', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def identify_category(self, filename: str) -> str:
        fn = filename.lower()
        if any(k in fn for k in ['is456', 'is13920', 'is875', 'sp16']):
            return 'engineering_standard'
        if any(k in fn for k in ['gcc', 'cpwd', 'contract']):
            return 'contract_gcc'
        if any(k in fn for k in ['pmgsy', 'hindi', 'pwd']):
            return 'policy_hindi'
        return 'general'

    def parse_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        doc = pymupdf.open(pdf_path)
        doc_name = os.path.basename(pdf_path)
        doc_category = self.identify_category(doc_name)
        chunks: List[DocumentChunk] = []

        current_clause_id = "General"
        current_clause_title = "Overview"

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            raw_text = doc[page_idx].get_text("text")
            cleaned_text = self.clean_text(raw_text)

            if not cleaned_text or len(cleaned_text) < 40:
                continue

            page_chunks = self._chunk_page(
                doc_name, doc_category, page_num, cleaned_text,
                current_clause_id, current_clause_title
            )

            if page_chunks:
                current_clause_id = page_chunks[-1].clause_id
                current_clause_title = page_chunks[-1].clause_title
                chunks.extend(page_chunks)

        return chunks

    def _chunk_page(
        self,
        doc_name: str,
        doc_category: str,
        page_num: int,
        text: str,
        initial_clause_id: str,
        initial_clause_title: str
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        lines = text.split("\n")

        current_clause = initial_clause_id
        current_title = initial_clause_title
        current_buffer: List[str] = []

        def flush_buffer(clause_id: str, title: str):
            buf_str = "\n".join(current_buffer).strip()
            if not buf_str or len(buf_str) < 30:
                return

            if len(buf_str) <= self.max_chunk_chars:
                segments = [buf_str]
            else:
                words = buf_str.split(" ")
                segments = []
                cur, cur_len = [], 0
                for w in words:
                    cur.append(w)
                    cur_len += len(w) + 1
                    if cur_len >= self.max_chunk_chars:
                        segments.append(" ".join(cur))
                        cur = cur[-25:]
                        cur_len = sum(len(x) + 1 for x in cur)
                if cur:
                    segments.append(" ".join(cur))

            for i, seg in enumerate(segments):
                if len(seg.strip()) < 30:
                    continue
                suffix = f"_{i}" if len(segments) > 1 else ""
                cid_clean = clause_id.replace(' ', '_')
                chunk_id = f"{doc_name}_p{page_num}_{cid_clean}{suffix}"
                citation = f"{doc_name}, Page {page_num}, {clause_id} ({title})"
                enriched = f"[{doc_name} | {clause_id}: {title} | Page {page_num}]\n{seg}"
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_name=doc_name,
                    doc_category=doc_category,
                    page_number=page_num,
                    clause_id=clause_id,
                    clause_title=title,
                    content=enriched,
                    full_citation=citation
                ))

        clause_pat = re.compile(r'^(?:CLAUSE|Clause|खंड|धारा)\s+([0-9A-Z\u0966-\u096F]+)(?:\s*[:\-\.]?\s*(.*))?$', re.IGNORECASE)
        section_pat = re.compile(r'^(\d+\.\d+(?:\.\d+)?)\s+([A-Za-z\u0900-\u097F].*)$')
        tbl_pat = re.compile(r'^(Table|सारणी)\s+([0-9A-Za-z\u0966-\u096F]+)(?:\s*[:\-\.]?\s*(.*))?$', re.IGNORECASE)

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # 1. Statutory numbered clauses (e.g. Clause 2, खंड 3)
            m = clause_pat.match(line_str)
            if m:
                flush_buffer(current_clause, current_title)
                current_buffer = []
                current_clause = f"Clause {m.group(1)}"
                current_title = m.group(2).strip() if m.group(2) else "Statutory Provision"
                current_buffer.append(line_str)
                continue

            # 2. Numbered technical sections (e.g. 5.1 Cement, 26.4 Cover)
            m = section_pat.match(line_str)
            if m:
                flush_buffer(current_clause, current_title)
                current_buffer = []
                current_clause = f"Clause {m.group(1)}"
                current_title = m.group(2).strip()
                current_buffer.append(line_str)
                continue

            # 3. Tables (e.g. Table 16, सारणी 2)
            m = tbl_pat.match(line_str)
            if m:
                flush_buffer(current_clause, current_title)
                current_buffer = []
                current_clause = f"Table {m.group(2)}"
                current_title = m.group(3).strip() if m.group(3) else "Technical Data"
                current_buffer.append(line_str)
                continue

            # 4. Universal standalone headings (short title line without terminal punctuation)
            if 3 <= len(line_str) <= 45 and not line_str.endswith(('.', '।', ',', ';', ':', '-')) and not line_str[0].isdigit():
                flush_buffer(current_clause, current_title)
                current_buffer = []
                current_clause = line_str
                current_title = "Section Heading"
                current_buffer.append(line_str)
                continue

            current_buffer.append(line_str)

        flush_buffer(current_clause, current_title)
        return chunks
