from dotenv import load_dotenv
load_dotenv()

from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import get_embeddings
from rag.vectorstore import create_vectorstore

# Step 1 - Load
docs = load_documents("docs/")

# Step 2 - Chunk
chunks = chunk_documents(docs)

# Step 3 - Embed
embeddings = get_embeddings()

# Step 4 - Store
vectorstore = create_vectorstore(chunks, embeddings)

# Step 5 - Test retrieval
query = "What is RAG?"
results = vectorstore.similarity_search(query, k=3)

print("\n🔍 Test Query Results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:200])
    print(f"Source: {doc.metadata}")