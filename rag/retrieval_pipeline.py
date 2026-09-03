import sys
from collections import defaultdict
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


def retrieve_and_answer(query, verbose=True):
    """Main function to retrieve documents and generate answer."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    outcome = get_retriever().retrieve(query.strip())
    relevant_docs = outcome.documents

    if not relevant_docs:
        return (
            "I could not retrieve enough constitutional evidence to answer "
            "this question reliably."
        )

    if verbose:
        print(f"User Query: {query}")
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

    # Enhanced prompt for better response generation
    system_prompt = """You are a constitutional law expert specializing in the Constitution of Nepal.

Your task is to provide detailed, well-structured answers based on the constitutional text provided.

FORMATTING RULES:
1. Start with the main Part and Article title (e.g., "📘 Part 7 – Federal Executive | Article 76 – Appointment of Prime Minister")
2. Break down the answer by Sub-articles, clearly labeled (e.g., "🔹 Sub-article (1)")
3. For each sub-article, list the clauses if they exist (e.g., "(a)", "(b)", "(c)")
4. Use the EXACT hierarchy from the constitution: Part → Article → Sub-article → Clause
5. If multiple articles are relevant, present each one separately with clear headers
6. Use emojis for visual clarity: 📘 for Parts, 🔹 for Sub-articles, • for clauses
7. Present sub-articles in numerical order (1, 2, 3, etc.)

CONTENT RULES:
1. Only use information from the provided constitutional text
2. Paraphrase the content clearly while maintaining legal accuracy
3. Do not claim that the Constitution is silent merely because a passage is missing or ambiguous. In that case say: "I could not retrieve enough constitutional evidence to answer this question reliably."
4. Always cite the exact Part, Article, and Sub-article numbers
5. Present ALL relevant sub-articles in order - don't skip any
6. Combine information from multiple chunks of the same sub-article if needed
7. Every legal claim must be supported by a Part, Article, or Sub-article present in the supplied text
8. Distinguish the direct answer from related provisions; do not present every retrieved provision as equally relevant

EXAMPLE FORMAT:
📘 Part X – [Part Name]
Article Y – [Article Title]

🔹 Sub-article (1)
As per Part X, Article Y, Sub-article (1):
(a) [Content of clause a]
(b) [Content of clause b]

🔹 Sub-article (2)
As per Part X, Article Y, Sub-article (2):
[Content if no clauses, or list clauses if they exist]
"""

    user_prompt = f"""Question: {query}

Constitutional Text:
{structured_context}

Please provide a comprehensive answer following the formatting rules. Include ALL relevant sub-articles in numerical order."""

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Define the messages for the model
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Invoke the model with the structured input
    result = model.invoke(messages)

    # Display the response
    if verbose:
        print("\n" + "=" * 60)
        print("--- ANSWER ---")
        print("=" * 60)

    print(result.content)
    return result.content


if __name__ == "__main__":
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How is the Prime Minister elected in Nepal?"

    retrieve_and_answer(query)
