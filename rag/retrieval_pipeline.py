import sys
from collections import defaultdict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

persistent_directory = "./db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


def format_document_with_metadata(doc):
    """Format a document with its metadata for better context."""
    metadata = doc.metadata
    parts = []

    if "part" in metadata:
        parts.append(f"📘 {metadata['part']}")
        if "part_name" in metadata and metadata["part_name"]:
            parts[-1] += f" – {metadata['part_name']}"

    if "article" in metadata:
        parts.append(f"Article {metadata['article'].replace('Article ', '')}")
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
                d.metadata.get("subarticle", ""),
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


def extract_key_terms(query):
    """Extract key terms from the query for better search."""
    # Common words to ignore
    stop_words = {
        "how",
        "what",
        "when",
        "where",
        "who",
        "why",
        "is",
        "are",
        "the",
        "a",
        "an",
        "in",
        "of",
        "to",
        "for",
        "and",
        "or",
        "nepal",
        "constitution",
    }

    words = query.lower().split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 3]

    return key_terms


def expand_query(original_query):
    """Expand query with synonyms and variations for better retrieval."""
    expansions = [original_query]

    # Extract key terms from the query
    key_terms = extract_key_terms(original_query)

    # Add variations with common term replacements
    replacements = {
        "elected": ["appointed", "selected", "chosen"],
        "rights": ["freedoms", "liberties", "entitlements"],
        "duties": ["responsibilities", "obligations"],
        "president": ["head of state"],
        "parliament": ["legislature", "house of representatives"],
        "government": ["executive", "administration"],
        "law": ["act", "legislation", "statute"],
        "citizen": ["national", "people"],
    }

    for original, alternatives in replacements.items():
        if original in original_query.lower():
            for alt in alternatives:
                expansions.append(original_query.lower().replace(original, alt))

    # Add a simplified version with just key terms
    if key_terms:
        expansions.append(" ".join(key_terms))

    # Topic-specific boosters to ensure key articles are found
    query_lower = original_query.lower()

    if "prime minister" in query_lower:
        expansions.extend(
            [
                "Article 76 Constitution Council Ministers",
                "President appoints Prime Minister House Representatives",
                "Prime Minister majority vote confidence",
                "Federal Executive Prime Minister appointment",
            ]
        )

    if "fundamental rights" in query_lower or (
        "rights" in query_lower and "citizen" in query_lower
    ):
        expansions.extend(
            [
                "Part 3 Fundamental Rights",
                "Article 16 right to live with dignity",
                "freedom expression assembly",
            ]
        )

    if "duties" in query_lower and "citizen" in query_lower:
        expansions.extend(
            ["Article 48 duties citizens", "responsibilities citizens Nepal"]
        )

    if "president" in query_lower and (
        "elect" in query_lower or "appoint" in query_lower
    ):
        expansions.extend(
            [
                "Article 62 election President",
                "Electoral College President Vice-President",
            ]
        )

    return list(set(expansions))  # Remove duplicates


def retrieve_and_answer(query, verbose=True):
    """Main function to retrieve documents and generate answer."""

    # Expand query for better retrieval
    query_variations = expand_query(query)

    if verbose:
        print(f"User Query: {query}")
        print(f"Query Variations: {query_variations[:5]}...")  # Show first 5
        print()

    # Retrieve documents for each query variation
    all_docs = []
    seen_ids = set()

    for q in query_variations:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        docs = retriever.invoke(q)

        # Deduplicate based on content
        for doc in docs:
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    # Extract key terms from query to identify relevant articles
    query_key_terms = extract_key_terms(query)

    # Check if we found key articles - fetch ALL their sub-articles for completeness
    key_articles_found = set()
    for doc in all_docs[:15]:  # Check top 15 docs
        article = doc.metadata.get("article")
        # If this document seems highly relevant (contains query terms), mark article as key
        if article:
            content_lower = doc.page_content.lower()
            # Count how many query terms appear in this document
            term_count = sum(1 for term in query_key_terms if term in content_lower)
            if (
                term_count >= 2 or len(query_key_terms) <= 1
            ):  # At least 2 terms or single-term query
                key_articles_found.add(article)

    # Fetch all chunks for key articles to ensure completeness
    if key_articles_found and verbose:
        print(f"Key articles detected: {key_articles_found}")
        print("Fetching complete articles for comprehensive answer...\n")

    complete_article_docs = []
    if key_articles_found:
        all_db_docs = db.get()
        for i, metadata in enumerate(all_db_docs["metadatas"]):
            if metadata and metadata.get("article") in key_articles_found:
                # Create a Document object
                from langchain_core.documents import Document

                doc = Document(
                    page_content=all_db_docs["documents"][i], metadata=metadata
                )
                # Check if not already in our list
                doc_id = doc.page_content[:100]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    complete_article_docs.append(doc)

    # Combine all documents
    all_docs.extend(complete_article_docs)

    # Filter and prioritize documents based on query relevance
    query_key_terms = extract_key_terms(query)
    priority_docs = []
    other_docs = []

    for doc in all_docs:
        content_lower = doc.page_content.lower()
        # Count how many query key terms appear in the document
        relevance_score = sum(1 for term in query_key_terms if term in content_lower)

        if relevance_score >= 1:  # At least one key term
            priority_docs.append((relevance_score, doc))
        else:
            other_docs.append(doc)

    # Sort priority docs by relevance score (descending), then by article/subarticle
    priority_docs.sort(
        key=lambda x: (
            -x[0],
            x[1].metadata.get("article", "ZZZ"),
            x[1].metadata.get("subarticle", "ZZZ"),
        )
    )
    priority_docs = [doc for score, doc in priority_docs]  # Remove scores

    # Combine with priority docs first, limit to top 20 for comprehensive coverage
    relevant_docs = priority_docs[:18] + other_docs[:2]

    if verbose:
        print(f"Total unique documents retrieved: {len(all_docs)}")
        print(f"Priority documents: {len(priority_docs)}\n")

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
3. If the constitution doesn't address the question, say: "The Constitution of Nepal does not address this question."
4. Always cite the exact Part, Article, and Sub-article numbers
5. Present ALL relevant sub-articles in order - don't skip any
6. Combine information from multiple chunks of the same sub-article if needed

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
