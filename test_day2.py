from dotenv import load_dotenv
load_dotenv()

from rag.embedder import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.retriever import create_qa_chain

# Load existing vectorstore
embeddings = get_embeddings()
vectorstore = load_vectorstore(embeddings)

# Create QA Chain
chain, retriever = create_qa_chain(vectorstore)

# Test Questions
questions = [
    "What is RAG?",
    "How does attention mechanism work?",
    "What are large language models?"
]

print("\n" + "="*50)
for question in questions:
    print(f"\n❓ Question: {question}")
    answer = chain.invoke(question)
    print(f"\n💡 Answer: {answer}")
    docs = retriever.invoke(question)
    print(f"\n📄 Sources:")
    for doc in docs:
        print(f"  - {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
    print("="*50)