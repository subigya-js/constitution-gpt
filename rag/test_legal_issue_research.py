import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from rag.retrieval_pipeline import (
    AnswerVerification,
    ConstitutionalAnswer,
    QueryScope,
    retrieve_and_answer,
)


def article(number, text):
    return Document(
        page_content=text,
        metadata={
            "part": "Part test",
            "article": f"Article {number}",
            "subarticle": "Sub-article (1)",
        },
    )


class LegalIssueResearchTests(unittest.TestCase):
    @patch("rag.retrieval_pipeline.verify_answer")
    @patch("rag.retrieval_pipeline.get_answer_model")
    @patch("rag.retrieval_pipeline.get_retriever")
    def test_eligibility_retrieves_each_issue_and_rejects_incomplete_answer(
        self, get_retriever, get_answer_model, verify_answer
    ):
        scope = QueryScope(
            category="constitutional",
            answer_type="eligibility",
            reason="Eligibility depends on multiple constitutional rules.",
            constitutional_query="Can an Indian citizen become Prime Minister of Nepal?",
            constitutional_queries=[
                "What citizenship status is required to become Prime Minister?",
                "What are the citizenship qualifications for House membership?",
                "How is a Prime Minister constitutionally appointed?",
            ],
            required_issues=[
                "Citizenship status required for Prime Minister",
                "House of Representatives membership qualifications",
                "Prime Minister appointment eligibility",
                "Special citizenship restrictions for constitutional offices",
            ],
            external_component="",
            recommended_source="",
            clarification_question="",
        )
        evidence = {
            scope.constitutional_queries[0]: [
                article(289, "The Prime Minister must have citizenship by descent.")
            ],
            scope.constitutional_queries[1]: [
                article(87, "A member must be a citizen of Nepal.")
            ],
            scope.constitutional_queries[2]: [
                article(76, "The President appoints the Prime Minister.")
            ],
            scope.constitutional_query: [],
        }

        def retrieve(query):
            return SimpleNamespace(
                documents=evidence[query],
                channel_counts={"semantic": 1},
                top_score=1.0,
            )

        get_retriever.return_value.retrieve.side_effect = retrieve
        get_answer_model.return_value.invoke.return_value = ConstitutionalAnswer(
            direct_answer="No. Article 76(1) controls the appointment.",
            primary_legal_basis="Article 76(1) concerns appointment.",
            supporting_sections=[],
            summary="The appointment is controlled by Article 76(1).",
            constitutional_evidence_sufficient=True,
        )
        verify_answer.return_value = AnswerVerification(
            grounded=True,
            citations_supported=True,
            issues_complete=False,
            injection_followed=False,
            unsupported_claims=[],
            missing_issues=["Citizenship status required for Prime Minister"],
            reason="The answer omitted a controlling eligibility issue.",
        )

        result = retrieve_and_answer(
            scope.constitutional_query,
            verbose=False,
            scope_override=scope,
        )

        called_queries = [
            call.args[0] for call in get_retriever.return_value.retrieve.call_args_list
        ]
        self.assertEqual(called_queries, [*scope.constitutional_queries, scope.constitutional_query])
        self.assertIn("sufficiently grounded", result)
        required_issues = verify_answer.call_args.args[3]
        self.assertIn("Citizenship status required for Prime Minister", required_issues)


if __name__ == "__main__":
    unittest.main()
