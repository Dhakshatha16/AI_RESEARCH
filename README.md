# 🤖 AI Research Agent

An intelligent research assistant that combines RAG with real-time web search.

## 🚀 Features
- 📄 Document Q&A using RAG
- 🌐 Real-time web search using Tavily
- 🤖 Auto mode - switches between RAG and web
- 💬 Chat interface using Streamlit

## 🛠️ Tech Stack
- LLM: Groq (Llama 3.3 70B)
- Embeddings: HuggingFace (MiniLM-L6-v2)
- Vector DB: ChromaDB
- Web Search: Tavily
- Framework: LangChain
- UI: Streamlit

## ⚙️ Setup
1. Clone the repo
2. pip install -r requirements.txt
3. Create .env with API keys
4. python test_day1.py
5. streamlit run app.py