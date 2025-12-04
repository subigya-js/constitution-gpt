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


def extract_part_name(text):
    """Extract Part number and name from text."""
    match = re.search(r'Part[-\s]*(\d+)\s*[-–—]?\s*([^\n]+)?', text, re.IGNORECASE)
    if match:
        part_num = match.group(1)
        part_name = match.group(2).strip() if match.group(2) else ""
        return f"Part {part_num}", part_name
    return None, None


def extract_article_info(text):
    """Extract Article number and title from text."""
    # Match patterns like "1. Article Title:" or just "1."
    match = re.search(r'^(\d+)\.\s*([^:\n]+)?', text.strip())
    if match:
        article_num = match.group(1)
        article_title = match.group(2).strip() if match.group(2) else ""
        return f"Article {article_num}", article_title
    return None, None


def extract_subarticle_num(text):
    """Extract sub-article number from text."""
    match = re.search(r'^\((\d+)\)', text.strip())
    if match:
        return f"Sub-article ({match.group(1)})"
    return None


def extract_clause_letter(text):
    """Extract clause letter from text."""
    match = re.search(r'^\(([a-z])\)', text.strip())
    if match:
        return f"Clause ({match.group(1)})"
    return None


def chunk_documents(documents):
    """
    Improved hierarchical chunking for Constitution of Nepal:
    - Keeps complete sub-articles together (with all their clauses)
    - Adds contextual prefix to each chunk for better retrieval
    - Preserves full metadata hierarchy
    """
    print("Chunking Started...")
    
    # Extract the raw text from the documents
    full_text = "\n".join([doc.page_content for doc in documents])
    
    final_chunks = []
    
    # Split by PART
    parts = re.split(r'(?=Part[-\s]*\d+)', full_text, flags=re.IGNORECASE)
    
    for part_text in parts:
        if not part_text.strip() or len(part_text.strip()) < 10:
            continue
            
        part_num, part_name = extract_part_name(part_text)
        
        # Split by ARTICLE (numbered items like "1.", "2.", etc.)
        articles = re.split(r'\n(?=\d+\.\s+[A-Z])', part_text)
        
        for article_text in articles:
            if not article_text.strip():
                continue
                
            article_num, article_title = extract_article_info(article_text)
            
            # Skip if no article number found (likely preamble or other content)
            if not article_num:
                continue
            
            # Split by SUB-ARTICLE "(1)", "(2)", etc.
            subarticles = re.split(r'(?=\(\d+\))', article_text)
            
            for subarticle_text in subarticles:
                if not subarticle_text.strip() or len(subarticle_text.strip()) < 15:
                    continue
                    
                subarticle_num = extract_subarticle_num(subarticle_text)
                
                # If this is a sub-article, keep it complete with all its clauses
                if subarticle_num:
                    # Build metadata
                    metadata = {}
                    if part_num:
                        metadata['part'] = part_num
                        if part_name:
                            metadata['part_name'] = part_name
                    if article_num:
                        metadata['article'] = article_num
                        if article_title:
                            metadata['article_title'] = article_title
                    metadata['subarticle'] = subarticle_num
                    
                    # Create hierarchical reference
                    hierarchy_parts = []
                    if part_num:
                        hierarchy_parts.append(part_num)
                    if article_num:
                        hierarchy_parts.append(article_num)
                    hierarchy_parts.append(subarticle_num)
                    metadata['hierarchy'] = " → ".join(hierarchy_parts)
                    
                    # Add contextual prefix for better retrieval
                    context_prefix = ""
                    if part_name and article_title:
                        context_prefix = f"[{part_name} - {article_title}]\n\n"
                    
                    # Clean the text content
                    clean_text = context_prefix + subarticle_text.strip()
                    
                    # If chunk is too large, split it but preserve metadata
                    if len(clean_text) > 2000:
                        # For very large sub-articles, split by clauses
                        clauses = re.split(r'(?=\([a-z]\))', subarticle_text)
                        
                        for clause_text in clauses:
                            if not clause_text.strip() or len(clause_text.strip()) < 10:
                                continue
                            
                            clause_letter = extract_clause_letter(clause_text)
                            clause_metadata = metadata.copy()
                            
                            if clause_letter:
                                clause_metadata['clause'] = clause_letter
                                clause_metadata['hierarchy'] = metadata['hierarchy'] + f" → {clause_letter}"
                            
                            clause_content = context_prefix + clause_text.strip()
                            
                            final_chunks.append({
                                'content': clause_content,
                                'metadata': clause_metadata
                            })
                    else:
                        final_chunks.append({
                            'content': clean_text,
                            'metadata': metadata
                        })
                else:
                    # This is the article header/title, create a chunk for it
                    if article_num and len(subarticle_text.strip()) > 20:
                        metadata = {}
                        if part_num:
                            metadata['part'] = part_num
                            if part_name:
                                metadata['part_name'] = part_name
                        if article_num:
                            metadata['article'] = article_num
                            if article_title:
                                metadata['article_title'] = article_title
                        
                        hierarchy_parts = []
                        if part_num:
                            hierarchy_parts.append(part_num)
                        if article_num:
                            hierarchy_parts.append(article_num)
                        metadata['hierarchy'] = " → ".join(hierarchy_parts)
                        
                        final_chunks.append({
                            'content': subarticle_text.strip(),
                            'metadata': metadata
                        })
    
    print(f"Chunking Completed. Total Chunks: {len(final_chunks)}")
    return final_chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store with metadata"""
    print("Creating embeddings and storing in local ChromaDB...")
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Convert chunks with metadata → Document objects
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk['content'],
            metadata=chunk['metadata']
        )
        documents.append(doc)
    
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
    
    # Display first few chunks with metadata
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content: {chunk['content'][:200]}...")
    
    print(f"\n... and {len(chunks) - 5} more chunks")
    
    collection = create_vector_store(chunks)
    print(collection)


if __name__ == "__main__":
    main()
