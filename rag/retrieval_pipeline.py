import sys
from collections import defaultdict
from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

try:
    from rag.chroma_connection import create_langchain_chroma
    from rag.hybrid_retrieval import (
        HybridConstitutionRetriever,
        subarticle_number,
    )
except ModuleNotFoundError:  # Support running this file directly.
    from chroma_connection import create_langchain_chroma
    from hybrid_retrieval import (
        HybridConstitutionRetriever,
        subarticle_number,
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Initialize external clients only when the first request arrives."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return create_langchain_chroma(embedding_model)


@lru_cache(maxsize=1)
def get_retriever():
    """Reuse the deduplicated corpus and lexical index across requests."""
    return HybridConstitutionRetriever(get_vector_store())


def format_document_with_metadata(doc):
    """Format a document with its metadata for better context."""
    metadata = doc.metadata
    parts = []

    if "part" in metadata:
        parts.append(f"📘 {metadata['part']}")
        if "part_name" in metadata and metadata["part_name"]:
            parts[-1] += f" – {metadata['part_name']}"

    if "article" in metadata:
        article_label = str(metadata["article"])
        parts.append(
            article_label
            if article_label.lower().startswith("article ")
            else f"Article {article_label}"
        )
        if "article_title" in metadata and metadata["article_title"]:
            parts[-1] += f" – {metadata['article_title']}"

    if "subarticle" in metadata:
        parts.append(metadata["subarticle"])

    if "clause" in metadata:
        parts.append(metadata["clause"])

    header = " | ".join(parts) if parts else "General Content"

    return f"{header}\n{doc.page_content}"


def group_docs_by_article(docs):
    """Group documents by their article for better organization."""
    grouped = defaultdict(list)

    for doc in docs:
        metadata = doc.metadata
        key = (
            metadata.get("part", "Unknown"),
            metadata.get("article", "Unknown"),
            metadata.get("article_title", ""),
        )
        grouped[key].append(doc)

    return grouped


def create_structured_context(docs):
    """Create a structured context from documents with metadata."""
    grouped = group_docs_by_article(docs)

    context_parts = []

    for (part, article, article_title), doc_list in grouped.items():
        # Header for this article
        header = f"\n{'=' * 60}\n"
        if part != "Unknown":
            header += f"📘 {part}"
            if article != "Unknown":
                header += f" | {article}"
                if article_title:
                    header += f" – {article_title}"
        header += f"\n{'=' * 60}"

        context_parts.append(header)

        # Sort documents by subarticle and clause
        sorted_docs = sorted(
            doc_list,
            key=lambda d: (
                subarticle_number(d) or 0,
                d.metadata.get("clause", ""),
            ),
        )

        for doc in sorted_docs:
            metadata = doc.metadata
            sub_parts = []

            if "subarticle" in metadata:
                sub_parts.append(f"  🔹 {metadata['subarticle']}")
            if "clause" in metadata:
                sub_parts.append(f"    • {metadata['clause']}")

            if sub_parts:
                context_parts.append("\n".join(sub_parts))

            # Indent the content
            content_lines = doc.page_content.split("\n")
            indented_content = "\n".join(["    " + line for line in content_lines])
            context_parts.append(indented_content)

    return "\n\n".join(context_parts)


class SupportingSection(BaseModel):
    heading: str = Field(
        description="A short, question-specific heading for supporting details."
    )
    content: str = Field(
        description="Relevant supporting explanation with exact constitutional citations."
    )


class QueryScope(BaseModel):
    category: Literal[
        "constitutional",
        "related_current",
        "mixed",
        "out_of_scope",
        "ambiguous",
    ] = Field(description="The information-source category required by the question.")
    reason: str = Field(
        description="A concise user-facing explanation for the classification."
    )
    constitutional_query: str = Field(
        description=(
            "A standalone constitutional question for retrieval. Preserve the original "
            "question when it is constitutional; otherwise extract its constitutional "
            "component. Use an empty string when no constitutional component exists."
        )
    )
    external_component: str = Field(
        description=(
            "The current, historical, statistical, or otherwise external information "
            "requested. Use an empty string when none exists."
        )
    )
    recommended_source: str = Field(
        description=(
            "A category of authoritative source appropriate for the external component, "
            "without inventing a URL. Use an empty string when not applicable."
        )
    )
    clarification_question: str = Field(
        description=(
            "One concise clarification question for ambiguous input. Use an empty string "
            "for every other category."
        )
    )


class ConstitutionalAnswer(BaseModel):
    direct_answer: str = Field(
        description=(
            "A direct 1-3 sentence answer. Begin with Yes or No for binary questions "
            "and cite the strongest Article, Sub-article, and Clause immediately."
        )
    )
    primary_legal_basis: str = Field(
        description=(
            "A concise explanation of the strongest controlling provision, including "
            "its exact Part, Article, Sub-article, and Clause when available."
        )
    )
    supporting_sections: list[SupportingSection] = Field(
        description=(
            "Only details needed to fully answer the question. Enumerate material "
            "qualifications, conditions, exceptions, grounds, duties, or procedural "
            "steps individually when the controlling provision contains a list. "
            "Exclude unrelated provisions. Use an empty list when no extra detail is needed."
        )
    )
    summary: str = Field(
        description="A concise final conclusion that does not introduce new claims."
    )
    constitutional_evidence_sufficient: bool = Field(
        description=(
            "Whether the supplied text supports the constitutional component. For "
            "related_current and mixed questions, do not evaluate support for the "
            "unavailable external component."
        )
    )


def render_current_information_notice(scope: QueryScope) -> str:
    notice = (
        "The Constitution of Nepal does not establish the requested current fact "
        "because information such as current officeholders can change over time. "
        "This assistant does not currently use a verified live-information source, "
        "so it cannot reliably provide that fact."
    )
    if scope.recommended_source.strip():
        notice += (
            f" For current information, consult {scope.recommended_source.strip()}."
        )
    return notice


def render_constitutional_answer(
    answer: ConstitutionalAnswer,
    scope: QueryScope | None = None,
) -> str:
    sections = []
    if scope and scope.category in {"related_current", "mixed"}:
        sections.append(render_current_information_notice(scope))

    if not answer.constitutional_evidence_sufficient:
        sections.append(
            "I could not retrieve enough constitutional evidence to answer "
            "this question reliably."
        )
        return "\n\n".join(sections)

    sections.append(answer.direct_answer.strip())
    sections.append(
        "## Constitutional basis\n\n" + answer.primary_legal_basis.strip()
    )
    for section in answer.supporting_sections:
        sections.append(f"## {section.heading.strip()}\n\n{section.content.strip()}")
    sections.append("## Summary\n\n" + answer.summary.strip())
    return "\n\n".join(section for section in sections if section.strip())


def render_scope_boundary(scope: QueryScope) -> str:
    if scope.category == "ambiguous":
        return scope.clarification_question.strip() or (
            "Could you clarify which constitutional issue you want to understand?"
        )

    explanation = scope.reason.strip() or (
        "This question requires information outside the Constitution of Nepal."
    )
    response = [explanation]
    response.append(
        "This assistant currently answers from the Constitution of Nepal and cannot "
        "reliably provide that external information."
    )
    if scope.recommended_source.strip():
        response.append(
            f"For an authoritative answer, consult {scope.recommended_source.strip()}."
        )
    return "\n\n".join(response)


@lru_cache(maxsize=1)
def get_query_router():
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    return model.with_structured_output(QueryScope, method="json_schema", strict=True)


def classify_query(query: str) -> QueryScope:
    messages = [
        SystemMessage(
            content="""Classify questions for a Constitution of Nepal assistant.

CATEGORIES:
- constitutional: Answerable from constitutional text, including interpretation, rights, institutions, qualifications, procedures, and explicit Article references.
- related_current: Closely related to a constitutional office or institution but asks for changeable current facts, such as the present officeholder. The Constitution may define the office but cannot establish the current fact.
- mixed: Contains both a constitutional question and a current, historical, statistical, or other external component.
- out_of_scope: Requires information unrelated to interpreting or explaining the Constitution.
- ambiguous: Too unclear to determine what constitutional or external information is requested.

Do not classify a difficult or unfamiliar constitutional question as out_of_scope. For related_current and mixed questions, rewrite only the constitutional component as a standalone retrieval query. When the question asks for a current officeholder, target how that office is appointed, elected, selected, or constitutionally established; do not broaden the rewrite to unrelated powers or qualifications. Never answer the question or supply the current fact during classification."""
        ),
        HumanMessage(content=f"Classify this question:\n\n{query}"),
    ]
    return get_query_router().invoke(messages)


def retrieve_and_answer(query, verbose=True):
    """Main function to retrieve documents and generate answer."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    normalized_query = query.strip()
    scope = classify_query(normalized_query)

    if scope.category in {"out_of_scope", "ambiguous"}:
        return render_scope_boundary(scope)

    retrieval_query = scope.constitutional_query.strip() or normalized_query
    outcome = get_retriever().retrieve(retrieval_query)
    relevant_docs = outcome.documents

    if not relevant_docs:
        return (
            "I could not retrieve enough constitutional evidence to answer "
            "this question reliably."
        )

    if verbose:
        print(f"User Query: {query}")
        print(f"Question category: {scope.category}")
        print(f"Retrieval Query: {retrieval_query}")
        print(f"Retrieval channels: {outcome.channel_counts}")
        print(f"Top retrieval score: {outcome.top_score:.3f}")
        print(f"Context documents: {len(relevant_docs)}\n")

        # Display results with metadata
        print("--- Context ---")
        for i, doc in enumerate(relevant_docs[:12], 1):  # Show first 12 for brevity
            print(f"\nDocument {i}:")
            print(format_document_with_metadata(doc))
            print()

        if len(relevant_docs) > 12:
            print(f"\n... and {len(relevant_docs) - 12} more documents\n")

    # Create structured context
    structured_context = create_structured_context(relevant_docs)

    # The structured schema enforces answer-first output independently of how
    # much parent-article context retrieval supplies.
    system_prompt = """You are a constitutional law expert specializing in the Constitution of Nepal.

Your task is to answer the user's actual question first and then explain the supporting constitutional law.

ANSWER ORDER — MANDATORY:
1. DIRECT ANSWER: Give the conclusion in 1-3 sentences before any heading or background. For a yes/no question, the first word must be "Yes" or "No". Cite the strongest controlling Article, Sub-article, and Clause in this opening when available.
2. PRIMARY LEGAL BASIS: Explain the provision that directly controls the answer.
3. SUPPORTING SECTIONS: Include only the qualifications, exceptions, procedures, or related provisions necessary to answer the question completely.
4. SUMMARY: Restate the conclusion concisely without adding new claims.

CONTENT RULES:
1. Only use information from the provided constitutional text
2. Paraphrase the content clearly while maintaining legal accuracy
3. Do not claim that the Constitution is silent merely because a passage is missing or ambiguous. In that case say: "I could not retrieve enough constitutional evidence to answer this question reliably."
4. Always cite the exact Part, Article, and Sub-article numbers
5. Every legal claim must be supported by a provision present in the supplied text
6. Retrieved context is evidence, not an output checklist. Do not explain unrelated sub-articles merely because they were supplied
7. Distinguish the direct answer from supplementary context
8. Preserve important qualifications, exceptions, deadlines, and conditions that materially affect the answer
9. For non-binary questions, begin with the most useful concise factual answer rather than "Yes" or "No"
10. When a controlling provision contains a list of qualifications, conditions, exceptions, grounds, duties, or procedural steps, enumerate each material item. Never replace the list with a vague phrase such as "including other requirements"
11. Include the Part number and title in the primary legal basis whenever they are present in the supplied text

SOURCE-BOUNDARY RULES:
1. For a related_current question, explicitly explain that the Constitution defines the office or process but does not identify the current fact because it can change over time
2. Never answer a current or external component from model memory
3. For a mixed question, clearly separate what the Constitution establishes from what requires a current external source
4. Mention the recommended authoritative source category for an external component, but do not invent a URL
5. Treat constitutional_evidence_sufficient as referring only to the constitutional component, not the unavailable current component
"""

    user_prompt = f"""Original question: {normalized_query}

Question category: {scope.category}
Constitutional retrieval question: {retrieval_query}
External component: {scope.external_component or "None"}
Recommended source: {scope.recommended_source or "None"}

Constitutional Text:
{structured_context}

Return an answer following the mandatory answer order. Select only provisions relevant to the question."""

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(
        ConstitutionalAnswer,
        method="json_schema",
        strict=True,
    )

    # Define the messages for the model
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Invoke the model with the structured input
    result = structured_model.invoke(messages)
    rendered_answer = render_constitutional_answer(result, scope)

    # Display the response
    if verbose:
        print("\n" + "=" * 60)
        print("--- ANSWER ---")
        print("=" * 60)

    print(rendered_answer)
    return rendered_answer


if __name__ == "__main__":
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How is the Prime Minister elected in Nepal?"

    retrieve_and_answer(query)
