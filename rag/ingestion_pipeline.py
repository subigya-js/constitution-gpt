import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Build an absolute path to the PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(BASE_DIR, "data", "Constitution_English.pdf")


def load_documents(doc_path=DOCUMENT_PATH):
    """Load the document from the data folder."""

    loader = PyMuPDFLoader(doc_path)
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No documents found at path: {doc_path}")

    print("Documents loaded:", len(documents))

    return documents

# Chunking


def chunk_documents(documents):
    """Hybrid Chunking:
    1. Rule-based split by PART and ARTICLE (legal structure)
    2. Semantic Chunking inside each article
    """
    print("Chunking Started...")

    # Extract the raw text from the documents
    full_text = "\n".join([doc.page_content for doc in documents])

    # --- Step 1: Split by PART headings ---
    part_pattern = r"(?=Part[-\s]*\d+)"
    parts = re.split(part_pattern, full_text, flags=re.IGNORECASE)

    final_chunks = []

    embeddings = OpenAIEmbeddings()

    # Recursive fallback chunker
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Semantic chunker
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold=92
    )

    # --- Process each PART ---
    for part in parts:
        if not part.strip():
            continue

        # Step 2: Split by ARTICLE inside each PART
        article_pattern = r"(?=\n?\s*\d+\.)"
        articles = re.split(article_pattern, part)

        for article in articles:
            if not article.strip():
                continue

            # For Sub-articles
            sub_article_pattern = r"(?=\(\d+\))"
            sub_articles = re.split(sub_article_pattern, article)

            for sub in sub_articles:
                if not sub.strip():
                    continue

                # For Clauses
                clause_pattern = r"(?=\([a-z]\))"
                clauses = re.split(clause_pattern, sub)

                for clause in clauses:
                    if not clause.strip():
                        continue

                # Apply Semantic Chunking on Final Section
                try:
                    s_chunks = semantic_splitter.split_text(clause)

                    for c in s_chunks:
                        if len(c) > 2000:
                            final_chunks.extend(
                                fallback_splitter.split_text(c))

                        else:
                            final_chunks.append(c)

                except Exception as e:
                    final_chunks.extend(fallback_splitter.split_text(clause))
        
    print("Chunking Completed. Total Chunks:", len(final_chunks))
    return final_chunks


def main():
    print("Main Function:")
    documents = load_documents()
    
    print("Chunks:\n")
    chunks = chunk_documents(documents)

    for chunk in chunks:
        print(chunk)

    # TODO: Add embedding and vector store logic here in the future.


if __name__ == "__main__":
    main()
