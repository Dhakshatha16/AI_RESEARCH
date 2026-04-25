import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

from rag.embedder import get_embeddings
from rag.vectorstore import load_vectorstore, create_vectorstore
from rag.retriever import create_qa_chain
from rag.web_search import create_web_search_tool
from rag.loader import load_documents
from rag.chunker import chunk_documents

# Page Config
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # Search Mode Toggle
    st.subheader("🔍 Search Mode")
    search_mode = st.radio(
        "Choose search mode:",
        ["🤖 Auto (RAG + Web)", "📄 Documents Only", "🌐 Web Only"]
    )

    st.divider()

    # PDF Upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Process Documents"):
            with st.spinner("Processing..."):
                # Save uploaded files
                os.makedirs("docs", exist_ok=True)
                for file in uploaded_files:
                    with open(f"docs/{file.name}", "wb") as f:
                        f.write(file.getbuffer())

                # Reload vectorstore
                docs = load_documents("docs/")
                chunks = chunk_documents(docs)
                embeddings = get_embeddings()
                create_vectorstore(chunks, embeddings)
                st.success(f"✅ {len(uploaded_files)} file(s) processed!")
                st.rerun()

    st.divider()

    # Search History
    st.subheader("🕐 Search History")
    if "history" in st.session_state and st.session_state.history:
        for i, q in enumerate(st.session_state.history[-5:]):
            st.caption(f"{i+1}. {q[:40]}...")
    else:
        st.caption("No history yet!")

    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# Load Models
@st.cache_resource
def load_models():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    chain, retriever = create_qa_chain(vectorstore)
    search_tool = create_web_search_tool()
    return chain, retriever, search_tool

chain, retriever, search_tool = load_models()

# Main Chat UI
st.title("🤖 AI Research Agent")
st.caption("Ask anything — I'll search documents or the web!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything..."):

    # Save to history
    st.session_state.history.append(prompt)

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get answer based on mode
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Web Only Mode
            if search_mode == "🌐 Web Only":
                web_results = search_tool.invoke(prompt)
                results = web_results.get("results", [])
                response = "🌐 **Web Search Results:**\n\n"
                for r in results:
                    response += f"**{r['title']}**\n{r['content']}\n\n"
                    response += f"🔗 [Source]({r['url']})\n\n---\n\n"

            # Documents Only Mode
            elif search_mode == "📄 Documents Only":
                rag_answer = chain.invoke(prompt)
                docs = retriever.invoke(prompt)
                sources = list(set([
                    f"{doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})"
                    for doc in docs
                ]))
                response = f"{rag_answer}\n\n📄 **Sources:**\n"
                for source in sources:
                    response += f"- {source}\n"

            # Auto Mode
            else:
                rag_answer = chain.invoke(prompt)
                if "don't have enough information" in rag_answer.lower():
                    web_results = search_tool.invoke(prompt)
                    results = web_results.get("results", [])
                    response = "🌐 **Web Search Results:**\n\n"
                    for r in results:
                        response += f"**{r['title']}**\n{r['content']}\n\n"
                        response += f"🔗 [Source]({r['url']})\n\n---\n\n"
                else:
                    docs = retriever.invoke(prompt)
                    sources = list(set([
                        f"{doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})"
                        for doc in docs
                    ]))
                    response = f"{rag_answer}\n\n📄 **Sources:**\n"
                    for source in sources:
                        response += f"- {source}\n"

            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })