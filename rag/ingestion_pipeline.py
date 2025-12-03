import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

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
    """
    Ultra-fast hybrid chunking for Constitution of Nepal:
    - Split by PART
    - Split by ARTICLE
    - Split by SUBARTICLE
    - Split by CLAUSE
    - Fallback to fixed-size splitter
    """

    print("Chunking Started...")

    # Extract the raw text from the documents
    full_text = "\n".join([doc.page_content for doc in documents])

    final_chunks = []

    # Recursive fallback chunker
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Level 1 — PART

    parts = re.split(r"(?=Part[-\s]*\d+)", full_text, flags=re.IGNORECASE)

    # --- Process each PART ---
    for part in parts:
        if not part.strip():
            continue

        # Step 2: ARTICLE: "1." , "2."
        articles = re.split(r"(?=\n?\s*\d+\.)", part)

        for article in articles:
            if not article.strip():
                continue

            # Step 3: SUB-ARTICLE "(1)", "(2)"
            subs = re.split(r"(?=\(\d+\))", article)

            for sub in subs:
                if not sub.strip():
                    continue

                # Step 4: CLAUSE "(a)", "(b)"
                clauses = re.split(r"(?=\([a-z]\))", sub)

                for clause in clauses:
                    if not clause.strip():
                        continue

                 # Fallback splitting (always keeps chunks within safe size)
                    small_chunks = fallback_splitter.split_text(clause)
                    final_chunks.extend(small_chunks)

    print("Chunking Completed. Total Chunks:", len(final_chunks))
    return final_chunks

# Create Embeddings and Vector DB


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in local ChromaDB...")

    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Convert plain strings → Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def main():
    print("Main Function:")
    documents = load_documents()

    print("Chunks:\n")
    chunks = chunk_documents(documents)

    for chunk in chunks:
        print(chunk)

    collection = create_vector_store(chunks)
    print(collection)

    # TODO: Add embedding and vector store logic here in the future.


if __name__ == "__main__":
    main()
