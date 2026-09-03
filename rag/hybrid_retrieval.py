"""Hybrid retrieval for structured constitutional documents.

The retriever combines semantic search, BM25-style lexical search, and exact
citation lookup. Results are fused, reranked, deduplicated, and expanded to
include sibling chunks from the most relevant articles.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_core.documents import Document


SEMANTIC_K = 24
LEXICAL_K = 24
SEED_K = 8
MAX_PARENT_ARTICLES = 3
MAX_CONTEXT_CHARS = 45_000
RRF_K = 60
CORPUS_PAGE_SIZE = 250

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ARTICLE_PATTERN = re.compile(
    r"\b(?:article|art\.?)[\s:#-]*(\d+)(?:\s*\(\s*(\d+)\s*\))?",
    re.IGNORECASE,
)
SUBARTICLE_PATTERN = re.compile(
    r"\b(?:sub[\s-]*article|clause)[\s:#-]*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "in", "is", "it", "nepal", "nepalese", "of", "on",
    "or", "that", "the", "this", "to", "under", "what", "when", "where",
    "which", "who", "why", "with",
}

# A compact legal vocabulary normalizes institutional names. These are concept
# aliases, not query-to-article rules; retrieval still decides the citation.
CONCEPT_ALIASES = {
    "parliament": "federal legislature house representatives national assembly",
    "lower house": "house representatives",
    "upper house": "national assembly",
}


@dataclass(frozen=True)
class Citation:
    article: int | None = None
    subarticle: int | None = None


@dataclass(frozen=True)
class RetrievalOutcome:
    documents: list[Document]
    channel_counts: dict[str, int]
    top_score: float


def _stem(token: str) -> str:
    """Apply conservative normalization without a language-model dependency."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS
    ]


def query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    query_lower = query.lower()
    for concept, aliases in CONCEPT_ALIASES.items():
        if concept in query_lower:
            tokens.extend(tokenize(aliases))
    return tokens


def extract_citation(query: str) -> Citation:
    article_match = ARTICLE_PATTERN.search(query)
    subarticle_match = SUBARTICLE_PATTERN.search(query)

    article = int(article_match.group(1)) if article_match else None
    inline_subarticle = (
        int(article_match.group(2))
        if article_match and article_match.group(2)
        else None
    )
    subarticle = (
        int(subarticle_match.group(1))
        if subarticle_match
        else inline_subarticle
    )
    return Citation(article=article, subarticle=subarticle)


def _metadata_number(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def article_number(document: Document) -> int | None:
    return _metadata_number(
        document.metadata.get("article_number", document.metadata.get("article"))
    )


def subarticle_number(document: Document) -> int | None:
    return _metadata_number(
        document.metadata.get(
            "subarticle_number", document.metadata.get("subarticle")
        )
    )


def document_key(document: Document) -> str:
    metadata = document.metadata
    identity = "|".join(
        [
            str(metadata.get("part", "")),
            str(article_number(document) or ""),
            str(subarticle_number(document) or ""),
            str(metadata.get("clause", "")),
            document.page_content.strip(),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def document_search_text(document: Document) -> str:
    metadata = document.metadata
    return " ".join(
        str(value)
        for value in (
            metadata.get("part_name", ""),
            metadata.get("article_title", ""),
            metadata.get("hierarchy", ""),
            document.page_content,
        )
        if value
    )


def deduplicate_documents(documents: Iterable[Document]) -> list[Document]:
    unique: dict[str, Document] = {}
    for document in documents:
        unique.setdefault(document_key(document), document)
    return list(unique.values())


class BM25Index:
    """Immutable lexical index built once and shared across requests."""

    def __init__(self, documents: Sequence[Document]):
        self.documents = list(documents)
        self.tokenized_documents = [
            tokenize(document_search_text(document)) for document in documents
        ]
        self.frequencies = [Counter(tokens) for tokens in self.tokenized_documents]
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokenized_documents:
            self.document_frequency.update(set(tokens))
        self.document_count = len(self.documents)
        self.average_length = sum(map(len, self.tokenized_documents)) / max(
            self.document_count, 1
        )

    def search(self, query: str, k: int = LEXICAL_K) -> list[Document]:
        query_terms = query_tokens(query)
        if not query_terms or not self.documents:
            return []

        scored: list[tuple[float, Document]] = []
        for document, tokens, frequencies in zip(
            self.documents, self.tokenized_documents, self.frequencies
        ):
            document_length = len(tokens)
            score = 0.0

            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                containing_documents = self.document_frequency[term]
                inverse_frequency = math.log(
                    1 + (self.document_count - containing_documents + 0.5)
                    / (containing_documents + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1 - 0.75
                    + 0.75 * document_length / max(self.average_length, 1)
                )
                score += inverse_frequency * frequency * 2.2 / denominator

            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:k]]


def bm25_search(
    query: str, documents: Sequence[Document], k: int = LEXICAL_K
) -> list[Document]:
    """Convenience wrapper used by tests and one-off callers."""
    return BM25Index(documents).search(query, k)


def metadata_search(query: str, documents: Sequence[Document]) -> list[Document]:
    """Resolve explicit legal citations without relying on similarity search."""
    citation = extract_citation(query)
    if citation.article is None:
        return []

    matches = [
        document
        for document in documents
        if article_number(document) == citation.article
        and (
            citation.subarticle is None
            or subarticle_number(document) == citation.subarticle
        )
    ]
    return sorted(matches, key=_constitutional_sort_key)


def reciprocal_rank_fusion(
    result_sets: Sequence[Sequence[Document]],
    rank_constant: int = RRF_K,
) -> tuple[list[Document], dict[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    documents_by_key: dict[str, Document] = {}

    for result_set in result_sets:
        seen_in_result_set: set[str] = set()
        for rank, document in enumerate(result_set, start=1):
            key = document_key(document)
            if key in seen_in_result_set:
                continue
            seen_in_result_set.add(key)
            documents_by_key.setdefault(key, document)
            scores[key] += 1 / (rank_constant + rank)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_key[key] for key in ranked_keys], dict(scores)


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    fusion_scores: dict[str, float],
) -> list[tuple[float, Document]]:
    """Rerank fused candidates with coverage, titles, and citation agreement."""
    query_terms = set(query_tokens(query))
    citation = extract_citation(query)
    reranked: list[tuple[float, Document]] = []

    for document in documents:
        body_terms = set(tokenize(document.page_content))
        title_terms = set(tokenize(str(document.metadata.get("article_title", ""))))
        coverage = len(query_terms & body_terms) / max(len(query_terms), 1)
        title_coverage = len(query_terms & title_terms) / max(len(query_terms), 1)
        citation_bonus = 0.0

        if citation.article is not None and article_number(document) == citation.article:
            citation_bonus += 3.0
        if (
            citation.subarticle is not None
            and subarticle_number(document) == citation.subarticle
        ):
            citation_bonus += 1.5

        fusion_score = fusion_scores.get(document_key(document), 0.0) * RRF_K
        score = fusion_score + coverage * 2.0 + title_coverage * 2.5 + citation_bonus
        reranked.append((score, document))

    reranked.sort(key=lambda item: item[0], reverse=True)
    return reranked


def _constitutional_sort_key(document: Document) -> tuple[int, int, str]:
    return (
        article_number(document) or 10_000,
        subarticle_number(document) or 0,
        str(document.metadata.get("clause", "")),
    )


def expand_parent_articles(
    seed_documents: Sequence[Document],
    corpus: Sequence[Document],
    max_articles: int = MAX_PARENT_ARTICLES,
) -> list[Document]:
    """Return complete sibling context for the highest-ranked articles."""
    selected_articles: list[int] = []
    for document in seed_documents:
        number = article_number(document)
        if number is not None and number not in selected_articles:
            selected_articles.append(number)
        if len(selected_articles) == max_articles:
            break

    siblings_by_article: defaultdict[int, list[Document]] = defaultdict(list)
    for document in corpus:
        number = article_number(document)
        if number in selected_articles:
            siblings_by_article[number].append(document)

    expanded: list[Document] = []
    per_article_budget = MAX_CONTEXT_CHARS // max(len(selected_articles), 1)
    for number in selected_articles:
        siblings = sorted(siblings_by_article[number], key=_constitutional_sort_key)
        article_seeds = [
            document
            for document in seed_documents
            if article_number(document) == number
        ]
        ordered_documents = deduplicate_documents([*article_seeds, *siblings])
        article_characters = 0

        for document in ordered_documents:
            content_size = len(document.page_content)
            if article_characters and article_characters + content_size > per_article_budget:
                break
            expanded.append(document)
            article_characters += content_size

    # Preserve relevant chunks without article metadata, if any.
    expanded.extend(doc for doc in seed_documents if article_number(doc) is None)
    return deduplicate_documents(expanded)


class HybridConstitutionRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._corpus: list[Document] | None = None
        self._lexical_index: BM25Index | None = None

    def corpus(self) -> list[Document]:
        if self._corpus is None:
            documents = []
            offset = 0
            while True:
                raw = self.vector_store.get(
                    include=["documents", "metadatas"],
                    limit=CORPUS_PAGE_SIZE,
                    offset=offset,
                )
                page = [
                    Document(page_content=content, metadata=metadata or {})
                    for content, metadata in zip(
                        raw.get("documents", []), raw.get("metadatas", [])
                    )
                    if content
                ]
                documents.extend(page)
                if len(page) < CORPUS_PAGE_SIZE:
                    break
                offset += CORPUS_PAGE_SIZE

            self._corpus = deduplicate_documents(documents)
            self._lexical_index = BM25Index(self._corpus)
        return self._corpus

    def retrieve(self, query: str) -> RetrievalOutcome:
        corpus = self.corpus()
        semantic = self.vector_store.similarity_search(query, k=SEMANTIC_K)
        lexical = self._lexical_index.search(query) if self._lexical_index else []
        metadata = metadata_search(query, corpus)

        fused, fusion_scores = reciprocal_rank_fusion(
            [metadata, lexical, semantic]
        )
        reranked = rerank_documents(query, fused, fusion_scores)
        seeds = [document for _, document in reranked[:SEED_K]]
        expanded = expand_parent_articles(seeds, corpus)

        return RetrievalOutcome(
            documents=expanded or seeds,
            channel_counts={
                "semantic": len(semantic),
                "lexical": len(lexical),
                "metadata": len(metadata),
            },
            top_score=reranked[0][0] if reranked else 0.0,
        )
