import unittest

from rag.retrieval_pipeline import (
    ConstitutionalAnswer,
    SupportingSection,
    render_constitutional_answer,
)


class AnswerFormattingTests(unittest.TestCase):
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
            evidence_sufficient=True,
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
            evidence_sufficient=False,
        )

        rendered = render_constitutional_answer(answer)
        self.assertEqual(
            rendered,
            "I could not retrieve enough constitutional evidence to answer this question reliably.",
        )


if __name__ == "__main__":
    unittest.main()
