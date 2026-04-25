from dotenv import load_dotenv
load_dotenv()

from rag.embedder import get_embeddings
from rag.vectorstore import load_vectorstore
from rag.retriever import create_qa_chain
from rag.web_search import create_web_search_tool

# Load vectorstore
embeddings = get_embeddings()
vectorstore = load_vectorstore(embeddings)

# Create QA Chain
chain, retriever = create_qa_chain(vectorstore)

# Create Web Search Tool
search_tool = create_web_search_tool()

def answer_question(question):
    print(f"\n❓ Question: {question}")

    # First try RAG
    rag_answer = chain.invoke(question)

    # Check if RAG has enough info
    if "don't have enough information" in rag_answer.lower():
        print("📡 RAG insufficient → Using Web Search...")
        web_results = search_tool.invoke(question)
        print(f"🌐 Web Answer: {web_results}")
    else:
        print(f"💡 RAG Answer: {rag_answer}")
        docs = retriever.invoke(question)
        print("📄 Sources:")
        for doc in docs:
            print(f"  - {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")

print("\n" + "="*50)

# Test 1 - Document question (RAG)
answer_question("What is RAG?")

print("="*50)

# Test 2 - Recent news (Web Search)
answer_question("What are the latest AI developments in 2025?")

print("="*50)

# Test 3 - Document question (RAG)
answer_question("How does attention mechanism work?")