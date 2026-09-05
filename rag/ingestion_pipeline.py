import hashlib
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from rag.chroma_connection import create_langchain_chroma, get_collection_name
    from rag.runtime_config import openai_client_options
except ModuleNotFoundError:  # Support running this file directly.
    from chroma_connection import create_langchain_chroma, get_collection_name
    from runtime_config import openai_client_options

load_dotenv()

# Build an absolute path to the PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENT_PATH = os.path.join(BASE_DIR, "data", "Constitution_English.pdf")


def load_documents(doc_path=DOCUMENT_PATH):
    """Load the document from the data folder."""
    loader = PyPDFLoader(doc_path)
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


def split_article_into_subarticles(article_text):
    """Split only monotonic sub-article markers, not numeric cross-references.

    Constitutional text can contain a line such as ``(1) shall be so held``
    inside sub-article (2). A plain regex split treats that reference as a new
    sub-article. Accepting only the next expected number preserves hierarchy.
    """
    candidates = list(
        re.finditer(r'\((\d+)\)\s*(?=[A-Z])', article_text)
    )
    boundaries = []
    expected_number = 1

    for candidate in candidates:
        number = int(candidate.group(1))
        if number == expected_number:
            boundaries.append(candidate.start())
            expected_number += 1

    if not boundaries:
        return [article_text]

    sections = []
    prefix = article_text[:boundaries[0]]
    if prefix.strip():
        sections.append(prefix)

    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(article_text)
        sections.append(article_text[start:end])

    return sections


def split_content_by_byte_limit(content, max_bytes=8000):
    """Split oversized cloud records on token boundaries under a byte limit."""
    if len(content.encode('utf-8')) <= max_bytes:
        return [content]

    segments = []
    current_tokens = []
    current_size = 0

    for token in re.findall(r'\S+\s*', content):
        token_size = len(token.encode('utf-8'))
        if current_tokens and current_size + token_size > max_bytes:
            segments.append(''.join(current_tokens).strip())
            current_tokens = []
            current_size = 0
        current_tokens.append(token)
        current_size += token_size

    if current_tokens:
        segments.append(''.join(current_tokens).strip())

    return segments


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
            
            # Split by actual sequential sub-articles while preserving numeric
            # cross-references that happen to begin a line.
            subarticles = split_article_into_subarticles(article_text)
            
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


def create_vector_store(chunks, persist_directory=None):
    """Create and persist ChromaDB vector store with metadata"""
    print("Creating embeddings and updating the configured Chroma collection...")
    
    # Explicit local paths are still supported for development and migration.
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    
    # Convert chunks with metadata → Document objects
    documents = []
    document_ids = []
    for chunk in chunks:
        content_segments = split_content_by_byte_limit(chunk['content'])
        for segment_index, content in enumerate(content_segments, start=1):
            metadata = chunk['metadata'].copy()
            article_match = re.search(r'\d+', str(metadata.get('article', '')))
            part_match = re.search(r'\d+', str(metadata.get('part', '')))
            subarticle_match = re.search(r'\d+', str(metadata.get('subarticle', '')))

            if article_match:
                metadata['article_number'] = int(article_match.group())
            if part_match:
                metadata['part_number'] = int(part_match.group())
            if subarticle_match:
                metadata['subarticle_number'] = int(subarticle_match.group())
            if len(content_segments) > 1:
                metadata['segment_number'] = segment_index

            parent_identity = (
                f"part-{metadata.get('part_number', 'unknown')}-"
                f"article-{metadata.get('article_number', 'unknown')}"
            )
            metadata['parent_article_id'] = parent_identity

            stable_identity = "|".join([
                parent_identity,
                str(metadata.get('subarticle_number', '')),
                str(metadata.get('clause', '')),
                content.strip(),
            ])
            chunk_id = hashlib.sha256(stable_identity.encode('utf-8')).hexdigest()
            metadata['chunk_id'] = chunk_id

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            document_ids.append(chunk_id)
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        **openai_client_options(),
    )
    
    print("--- Creating vector store ---")
    vectorstore = create_langchain_chroma(
        embedding_function=embedding_model,
        local_path=persist_directory,
    )
    batch_size = int(os.getenv("CHROMA_UPSERT_BATCH_SIZE", "250"))
    if batch_size < 1:
        raise ValueError("CHROMA_UPSERT_BATCH_SIZE must be at least 1")
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        vectorstore.add_documents(
            documents[start:end],
            ids=document_ids[start:end],
        )
        print(f"Uploaded {end}/{len(documents)} chunks")
    
    print("--- Finished creating vector store ---")
    print(f"Vector store updated in collection: {get_collection_name()}")
    
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
