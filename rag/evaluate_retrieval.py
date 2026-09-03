"""Run labeled retrieval evaluation without invoking the answer model."""

import json
from pathlib import Path

try:
    from rag.hybrid_retrieval import article_number, subarticle_number
    from rag.retrieval_pipeline import get_retriever
except ModuleNotFoundError:  # Support running this file directly.
    from hybrid_retrieval import article_number, subarticle_number
    from retrieval_pipeline import get_retriever


CASES_PATH = Path(__file__).with_name("retrieval_eval_cases.json")


def evaluate():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    retriever = get_retriever()
    article_hits = 0
    subarticle_hits = 0
    subarticle_cases = 0

    for case in cases:
        outcome = retriever.retrieve(case["query"])
        references = {
            (article_number(document), subarticle_number(document))
            for document in outcome.documents
        }
        article_hit = any(
            article == case["expected_article"] for article, _ in references
        )
        expected_subarticle = case.get("expected_subarticle")
        subarticle_hit = (
            expected_subarticle is None
            or (case["expected_article"], expected_subarticle) in references
        )

        article_hits += int(article_hit)
        if expected_subarticle is not None:
            subarticle_cases += 1
            subarticle_hits += int(subarticle_hit)

        status = "PASS" if article_hit and subarticle_hit else "FAIL"
        print(
            f"{status}: {case['query']} "
            f"-> expected Article {case['expected_article']}"
        )

    article_recall = article_hits / len(cases)
    subarticle_recall = subarticle_hits / max(subarticle_cases, 1)
    print(f"\nArticle recall: {article_recall:.1%}")
    print(f"Sub-article recall: {subarticle_recall:.1%}")

    return article_recall, subarticle_recall


if __name__ == "__main__":
    evaluate()
