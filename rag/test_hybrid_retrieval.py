import unittest

from langchain_core.documents import Document

from rag.hybrid_retrieval import (
    HybridConstitutionRetriever,
    article_number,
    bm25_search,
    deduplicate_documents,
    extract_citation,
    metadata_search,
)
from rag.ingestion_pipeline import (
    split_article_into_subarticles,
    split_content_by_byte_limit,
)


def constitutional_document(article, subarticle, title, content):
    return Document(
        page_content=content,
        metadata={
            "part": "Part 8",
            "part_name": "Federal Legislature",
            "article": f"Article {article}",
            "article_title": title,
            "subarticle": f"Sub-article ({subarticle})",
            "hierarchy": f"Part 8 → Article {article} → Sub-article ({subarticle})",
        },
    )


ARTICLE_91_1 = constitutional_document(
    91,
    1,
    "Speaker and Deputy Speaker of House of Representatives",
    "The members of the House of Representatives shall elect a Speaker and a "
    "Deputy Speaker from amongst themselves.",
)
ARTICLE_91_6 = constitutional_document(
    91,
    6,
    "Speaker and Deputy Speaker of House of Representatives",
    "The office of the Speaker or Deputy Speaker shall become vacant in the "
    "following circumstances.",
)
UNRELATED = constitutional_document(
    62,
    1,
    "Election of President",
    "The President shall be elected by an electoral college.",
)
ARTICLE_83 = constitutional_document(
    83,
    1,
    "Federal Legislature",
    "There shall be a Federal Legislature consisting of two Houses, namely "
    "the House of Representatives and the National Assembly.",
)


class FakeVectorStore:
    def __init__(self, corpus, semantic_results):
        self._corpus = corpus
        self._semantic_results = semantic_results

    def get(self, include=None, limit=None, offset=None):
        start = offset or 0
        end = start + limit if limit is not None else None
        corpus = self._corpus[start:end]
        return {
            "documents": [document.page_content for document in corpus],
            "metadatas": [document.metadata for document in corpus],
        }

    def similarity_search(self, query, k):
        return self._semantic_results[:k]


class HybridRetrievalTests(unittest.TestCase):
    def test_cloud_document_split_respects_byte_limit(self):
        segments = split_content_by_byte_limit("नागरिक अधिकार " * 100, max_bytes=100)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment.encode("utf-8")) <= 100 for segment in segments))

    def test_subarticle_parser_does_not_split_numeric_cross_reference(self):
        article = """91. Speaker and Deputy Speaker: (1) Members shall elect a Speaker.
(2) Election under clause
(1) shall include one woman.
(3) A vacancy shall be filled by election.
"""
        sections = split_article_into_subarticles(article)

        self.assertEqual(len(sections), 4)
        self.assertIn("(1) shall include one woman", sections[2])
        self.assertTrue(sections[3].lstrip().startswith("(3)"))

    def test_extracts_explicit_article_and_subarticle(self):
        citation = extract_citation("What does Article 91(6) say?")
        self.assertEqual(citation.article, 91)
        self.assertEqual(citation.subarticle, 6)

    def test_metadata_search_resolves_exact_citation(self):
        matches = metadata_search(
            "Explain Article 91, sub-article 6",
            [ARTICLE_91_1, ARTICLE_91_6, UNRELATED],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(article_number(matches[0]), 91)
        self.assertEqual(matches[0].metadata["subarticle"], "Sub-article (6)")

    def test_lexical_search_finds_speaker_without_topic_rules(self):
        results = bm25_search(
            "How is the speaker of parliament elected in Nepal?",
            [UNRELATED, ARTICLE_91_1, ARTICLE_91_6],
        )
        self.assertEqual(article_number(results[0]), 91)
        self.assertEqual(results[0].metadata["subarticle"], "Sub-article (1)")

    def test_concept_vocabulary_bridges_parliament_and_legislature(self):
        results = bm25_search(
            "What bodies make up the Federal Parliament?",
            [UNRELATED, ARTICLE_83],
        )
        self.assertEqual(article_number(results[0]), 83)

    def test_hybrid_retrieval_recovers_from_bad_semantic_result(self):
        store = FakeVectorStore(
            [UNRELATED, ARTICLE_91_1, ARTICLE_91_6, ARTICLE_91_1],
            semantic_results=[UNRELATED],
        )
        outcome = HybridConstitutionRetriever(store).retrieve(
            "How is the speaker of parliament elected in Nepal?"
        )

        article_91 = [
            document
            for document in outcome.documents
            if article_number(document) == 91
        ]
        self.assertEqual(len(article_91), 2)
        self.assertEqual(outcome.channel_counts["lexical"], 3)

    def test_duplicate_ingestions_do_not_duplicate_context(self):
        unique = deduplicate_documents([ARTICLE_91_1, ARTICLE_91_1, ARTICLE_91_6])
        self.assertEqual(len(unique), 2)


if __name__ == "__main__":
    unittest.main()
