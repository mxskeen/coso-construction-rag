import unittest
import os
import numpy as np
from src.parser import ConstructionDocumentParser
from src.fine_tuning import load_construction_embedder
from src.hybrid_retriever import ConstructionHybridRetriever
from src.generator import GroundedGenerator
from src.evaluate import is_clause_match


class TestConstructionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = ConstructionDocumentParser()
        cls.retriever = ConstructionHybridRetriever(use_finetuned=True)
        base = "data/corpus" if os.path.exists("data/corpus") else "RAG-Interview-Test-Corpus-20261005T065046Z-1-001/RAG-Interview-Test-Corpus"
        pdf_paths = [os.path.join(base, "PMGSY-III-summary_HINDI_selectable-text.pdf")]
        cls.retriever.build_index(pdf_paths)
        cls.generator = GroundedGenerator()

    def test_parser_extracts_metadata(self):
        base = "data/corpus" if os.path.exists("data/corpus") else "RAG-Interview-Test-Corpus-20261005T065046Z-1-001/RAG-Interview-Test-Corpus"
        chunks = self.parser.parse_pdf(os.path.join(base, "PMGSY-III-summary_HINDI_selectable-text.pdf"))
        self.assertGreater(len(chunks), 0)
        chunk = chunks[0]
        self.assertTrue(hasattr(chunk, "clause_id"))
        self.assertTrue(hasattr(chunk, "page_number"))
        self.assertTrue(hasattr(chunk, "full_citation"))
        self.assertGreater(len(chunk.content), 20)

    def test_embedder_dimension_and_normalization(self):
        model = load_construction_embedder(use_finetuned=True)
        vecs = model.encode(["Concrete test", "CPWD delay"], normalize_embeddings=True)
        self.assertEqual(vecs.shape[1], 384)
        norm = np.linalg.norm(vecs[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_hybrid_retrieval(self):
        results = self.retriever.retrieve("PMGSY", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("full_citation", results[0])
        self.assertIn("rrf_score", results[0])

    def test_generator_output_structure(self):
        sample_chunk = [{
            "doc_name": "IS456-2000.pdf",
            "clause_id": "Clause 26.4",
            "clause_title": "Nominal Cover",
            "page_number": 60,
            "full_citation": "IS456-2000.pdf, Page 60, Clause 26.4 (Nominal Cover)",
            "content": "For severe exposure, nominal cover is 45 mm."
        }]
        res = self.generator.generate("What is the nominal cover?", sample_chunk)
        self.assertIn("answer", res)
        self.assertIn("citations", res)
        self.assertEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["clause_id"], "Clause 26.4")

    def test_clause_matcher(self):
        self.assertTrue(is_clause_match("Clause 2", "Clause 2"))
        self.assertTrue(is_clause_match("Table 16", "Table 16"))
        self.assertTrue(is_clause_match("वित्तीय हिस्सेदारी", "वित्तीय हिस्सेदारी"))


if __name__ == "__main__":
    unittest.main()
