"""Deterministic controls for untrusted prompts, evidence, and model output.

These controls intentionally sit outside the LLM. Prompt instructions alone are
not a security boundary: every model input and output must be treated as
untrusted data and validated before it crosses the next boundary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from langchain_core.documents import Document

try:
    from rag.hybrid_retrieval import article_number, subarticle_number
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hybrid_retrieval import article_number, subarticle_number


RiskLevel = Literal["low", "medium", "high"]

SECURITY_REFUSAL = (
    "Nice try, Diddy. Bro, please be genuine and only ask a real question about the "
    "Constitution of Nepal. ⚖️"
)

_SECURITY_LOGGER = logging.getLogger("constitution_gpt.security")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# These patterns are deliberately narrow. A heuristic detector is a useful
# signal, not an authorization system, and broad keyword matching creates false
# positives for legitimate legal questions.
_RISK_PATTERNS: tuple[tuple[str, int, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        4,
        re.compile(
            r"\b(?:ignore|disregard|forget|bypass|override)\b.{0,80}"
            r"\b(?:previous|prior|above|system|developer|hidden|safety)\b.{0,40}"
            r"\b(?:instruction|prompt|rule|message|policy)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "prompt_disclosure",
        5,
        re.compile(
            r"\b(?:reveal|show|print|repeat|reproduce|quote|return|expose|leak|"
            r"share|provide|disclose|display|output)\b"
            r".{0,100}\b(?:system prompt|developer message|hidden (?:system )?instruction|"
            r"internal instruction|classification rule|prompt schema)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "prompt_discovery",
        5,
        re.compile(
            r"\b(?:what|where)\b.{0,30}"
            r"\b(?:system prompt|developer message|hidden (?:system )?instruction|internal instruction)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_impersonation",
        3,
        re.compile(
            r"(?:^|[\n\r])\s*(?:system|developer|assistant)\s*:|"
            r"<(?:system|developer|assistant)>|\[(?:system|developer|assistant)\]",
            re.IGNORECASE,
        ),
    ),
    (
        "privilege_claim",
        3,
        re.compile(
            r"\b(?:developer|administrator|root|debug|diagnostic|maintenance)\s+mode\b|"
            r"\b(?:higher|highest)\s+priority\s+(?:instruction|rule)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "encoded_instruction",
        4,
        re.compile(
            r"\b(?:base\s*64|rot\s*13|hex|decode|encoded?)\b.{0,100}"
            r"\b(?:prompt|instruction|system|secret|policy)\b|"
            r"\b(?:prompt|instruction|system|secret|policy)\b.{0,100}"
            r"\b(?:base\s*64|rot\s*13|hex|decode|encoded?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "fake_authority",
        4,
        re.compile(
            r"\b(?:new|updated|replacement)\s+(?:system\s+)?instructions?\b|"
            r"\btreat\s+(?:the\s+)?following\s+as\s+(?:an?\s+)?(?:amendment|instruction|policy)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class SecurityAssessment:
    risk_level: RiskLevel
    score: int
    signals: tuple[str, ...]
    fingerprint: str
    normalized_text: str


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    cited_articles: tuple[int, ...]
    unsupported_articles: tuple[int, ...]
    unsupported_subarticles: tuple[tuple[int, int], ...]
    has_primary_citation: bool


def security_refusal() -> str:
    """Return the fixed branded refusal without revealing detector details."""

    return SECURITY_REFUSAL


def normalize_untrusted_text(value: str) -> str:
    """Canonicalize text before classification and comparison.

    NFKC normalization and removal of zero-width controls close common detector
    bypasses without trying to reinterpret arbitrary user encodings.
    """

    normalized = unicodedata.normalize("NFKC", value).translate(_ZERO_WIDTH)
    return re.sub(r"[\t\r\f\v ]+", " ", normalized).strip()


def assess_prompt(value: str) -> SecurityAssessment:
    normalized = normalize_untrusted_text(value)
    score = 0
    signals: list[str] = []
    for name, weight, pattern in _RISK_PATTERNS:
        if pattern.search(normalized):
            score += weight
            signals.append(name)

    risk_level: RiskLevel
    if score >= 4:
        risk_level = "high"
    elif score >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    return SecurityAssessment(
        risk_level=risk_level,
        score=score,
        signals=tuple(signals),
        fingerprint=hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
        normalized_text=normalized,
    )


def extracted_task_is_safe(value: str) -> bool:
    """A router-produced task must contain no remaining control instructions."""

    normalized = normalize_untrusted_text(value)
    if not normalized or len(normalized) > 1000:
        return False
    return assess_prompt(normalized).risk_level == "low"


def filter_suspicious_documents(
    documents: Sequence[Document],
) -> tuple[list[Document], list[str]]:
    """Exclude retrieved chunks that look like indirect prompt injection.

    Constitution text is expected to be trusted today, but this protects the
    generation boundary if the index later ingests uploaded or remote content.
    """

    safe: list[Document] = []
    blocked_fingerprints: list[str] = []
    for document in documents:
        assessment = assess_prompt(document.page_content)
        if assessment.risk_level == "high":
            blocked_fingerprints.append(assessment.fingerprint)
        else:
            safe.append(document)
    return safe, blocked_fingerprints


_ARTICLE_PATTERN = re.compile(
    r"(?<!Sub-)(?<!Sub )\bArticle\s+(\d+)\b", re.IGNORECASE
)
_INLINE_SUBARTICLE_PATTERN = re.compile(
    r"(?<!Sub-)(?<!Sub )\bArticle\s+(\d+)\s*\(\s*(\d+)\s*\)",
    re.IGNORECASE,
)
_NAMED_SUBARTICLE_PATTERN = re.compile(
    r"(?<!Sub-)(?<!Sub )\bArticle\s+(\d+)\b.{0,120}?"
    r"\bSub[- ]?article\s*\(?\s*(\d+)\s*\)?",
    re.IGNORECASE | re.DOTALL,
)
_EVIDENCE_SUBARTICLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Sub[- ]?article\s*)?\(\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def validate_answer_citations(
    answer: str,
    documents: Iterable[Document],
    *,
    evidence_required: bool,
) -> CitationValidation:
    """Reject article/sub-article references absent from retrieved evidence."""

    evidence_articles: set[int] = set()
    evidence_pairs: set[tuple[int, int]] = set()
    for document in documents:
        article = article_number(document)
        if article is None:
            continue
        evidence_articles.add(article)
        subarticle = subarticle_number(document)
        if subarticle is not None:
            evidence_pairs.add((article, subarticle))
        # Parent-article chunks may contain numbered sub-articles even when the
        # ingestion metadata describes only the article.
        evidence_pairs.update(
            (article, int(value))
            for value in _EVIDENCE_SUBARTICLE_PATTERN.findall(
                document.page_content
            )
        )

    cited_articles = {int(value) for value in _ARTICLE_PATTERN.findall(answer)}
    cited_pairs = {
        (int(article), int(subarticle))
        for article, subarticle in (
            _INLINE_SUBARTICLE_PATTERN.findall(answer)
            + _NAMED_SUBARTICLE_PATTERN.findall(answer)
        )
    }
    unsupported_articles = cited_articles - evidence_articles
    unsupported_pairs = {
        pair
        for pair in cited_pairs
        if pair[0] in evidence_articles and pair not in evidence_pairs
    }

    first_section = answer.split("##", maxsplit=1)[0]
    primary_articles = {
        int(value) for value in _ARTICLE_PATTERN.findall(first_section)
    }
    has_primary_citation = bool(primary_articles & evidence_articles)
    valid = (
        not unsupported_articles
        and not unsupported_pairs
        and (not evidence_required or has_primary_citation)
    )

    return CitationValidation(
        valid=valid,
        cited_articles=tuple(sorted(cited_articles)),
        unsupported_articles=tuple(sorted(unsupported_articles)),
        unsupported_subarticles=tuple(sorted(unsupported_pairs)),
        has_primary_citation=has_primary_citation,
    )


def output_is_safe(value: str, canary: str) -> bool:
    if canary and canary in value:
        return False
    return assess_prompt(value).risk_level != "high"


def log_security_event(
    event: str,
    assessment: SecurityAssessment | None = None,
    **details: object,
) -> None:
    """Emit structured metadata without storing the raw user prompt."""

    payload: dict[str, object] = {"event": event, **details}
    if assessment is not None:
        payload.update(
            {
                "prompt_fingerprint": assessment.fingerprint,
                "risk_level": assessment.risk_level,
                "signals": assessment.signals,
            }
        )
    _SECURITY_LOGGER.warning(json.dumps(payload, sort_keys=True, default=str))
