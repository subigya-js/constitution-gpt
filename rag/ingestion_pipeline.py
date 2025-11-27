import os
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
    print("Documents loaded:", len(documents))
    return documents


def main():
    print("Main Function:")
    documents = load_documents()

    # TODO: Add embedding and vector store logic here in the future.


if __name__ == "__main__":
    main()
