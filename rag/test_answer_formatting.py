import unittest

from rag.retrieval_pipeline import (
    ConstitutionalAnswer,
    QueryScope,
    SupportingSection,
    render_constitutional_answer,
    render_current_information_notice,
    render_scope_boundary,
)


class AnswerFormattingTests(unittest.TestCase):
    def test_related_current_notice_is_deterministic(self):
        scope = QueryScope(
            category="related_current",
            reason="The current officeholder changes over time.",
            constitutional_query="How is the Prime Minister appointed?",
            external_component="Identity of the current Prime Minister",
            recommended_source="an official Government of Nepal source",
            clarification_question="",
        )
        notice = render_current_information_notice(scope)

        self.assertIn("does not establish the requested current fact", notice)
        self.assertIn("does not currently use a verified", notice)
        self.assertIn("official Government of Nepal", notice)

    def test_related_current_notice_precedes_constitutional_answer(self):
        scope = QueryScope(
            category="related_current",
            reason="The current officeholder changes over time.",
            constitutional_query="How is the Prime Minister appointed?",
            external_component="Identity of the current Prime Minister",
            recommended_source="an official Government of Nepal source",
            clarification_question="",
        )
        answer = ConstitutionalAnswer(
            direct_answer="The President appoints the Prime Minister under Article 76.",
            primary_legal_basis="Article 76 establishes the appointment process.",
            supporting_sections=[],
            summary="Article 76 controls the appointment process.",
            constitutional_evidence_sufficient=True,
        )

        rendered = render_constitutional_answer(answer, scope)
        self.assertTrue(rendered.startswith("The Constitution of Nepal does not"))
        self.assertLess(
            rendered.index("requested current fact"),
            rendered.index("The President appoints"),
        )

    def test_out_of_scope_response_explains_boundary_and_next_source(self):
        scope = QueryScope(
            category="out_of_scope",
            reason="Current economic statistics are not contained in the Constitution.",
            constitutional_query="",
            external_component="Current GDP",
            recommended_source="an official Government of Nepal statistics source",
            clarification_question="",
        )

        rendered = render_scope_boundary(scope)

        self.assertIn("not contained in the Constitution", rendered)
        self.assertIn("cannot reliably provide", rendered)
        self.assertIn("official Government of Nepal", rendered)

    def test_ambiguous_response_returns_clarification(self):
        scope = QueryScope(
            category="ambiguous",
            reason="The requested subject is unclear.",
            constitutional_query="",
            external_component="",
            recommended_source="",
            clarification_question="Which constitutional office do you mean?",
        )
        self.assertEqual(
            render_scope_boundary(scope),
            "Which constitutional office do you mean?",
        )

    def test_direct_answer_precedes_constitutional_details(self):
        answer = ConstitutionalAnswer(
            direct_answer=(
                "No. Under Article 87(1)(a), a member of the Federal Parliament "
                "must be a citizen of Nepal."
            ),
            primary_legal_basis=(
                "Part 8, Article 87(1)(a) makes Nepalese citizenship mandatory."
            ),
            supporting_sections=[
                SupportingSection(
                    heading="Other qualifications",
                    content="Article 87(1)(b)-(e) provides additional requirements.",
                )
            ],
            summary="A non-Nepali citizen is not qualified for membership.",
            constitutional_evidence_sufficient=True,
        )

        rendered = render_constitutional_answer(answer)

        self.assertTrue(rendered.startswith("No."))
        self.assertLess(rendered.index("No."), rendered.index("## Constitutional basis"))
        self.assertLess(rendered.index("## Constitutional basis"), rendered.index("## Summary"))

    def test_insufficient_evidence_does_not_render_unsupported_sections(self):
        answer = ConstitutionalAnswer(
            direct_answer="No supported answer is available.",
            primary_legal_basis="None.",
            supporting_sections=[],
            summary="No conclusion.",
            constitutional_evidence_sufficient=False,
        )

        rendered = render_constitutional_answer(answer)
        self.assertEqual(
            rendered,
            "I could not retrieve enough constitutional evidence to answer this question reliably.",
        )


if __name__ == "__main__":
    unittest.main()
