from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "./db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant document
query = "How is the Prime Minister elected in Nepal?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)
print(f"User Query: {query}\n")

# Display results
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine query adn relevant document contents
combine_input = f"""Based on the following documents, please answer this question: {query}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, please say "The constitution of Nepal does not address this question."
"""

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the messages from the model
messages = [
    SystemMessage(
        content="You are a helpful assistant who knows about the Constitution of Nepal. Please also mention the Part, Article, Sub-Article number(s) in your answer. Start the answer with 'As per the Part X, Article Y, Sub-Article Z of the Constitution of Nepal...'. The Part is in the form of Part-1, Article in the form of 1. , 2. , etc and Sub-Article in the form of (1), (2), etc. If there are multiple references, mention all of them. For example: If the answer refers to Part-2, Article 5, Sub-Article (1) and Part-3, Article 10, Sub-Article (2), mention both in the answer."),
    HumanMessage(content=combine_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generate Response ---")
print("Content Only:")
print(result.content)
