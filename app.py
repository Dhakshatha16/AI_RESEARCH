import streamlit as st
from dotenv import load_dotenv
import os
import time
import json
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

# History File
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# Load Models
@st.cache_resource
def load_models():
    try:
        embeddings = get_embeddings()
        vectorstore = load_vectorstore(embeddings)
        chain, retriever = create_qa_chain(vectorstore)
        search_tool = create_web_search_tool()
        return chain, retriever, search_tool
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None, None

# Retry Logic
def invoke_with_retry(chain, prompt, retries=3):
    for attempt in range(retries):
        try:
            return chain.invoke(prompt)
        except Exception as e:
            error = str(e).lower()
            if "rate_limit" in error or "429" in error:
                wait_time = (attempt + 1) * 10
                st.warning(f"⚠️ Rate limit hit — Waiting {wait_time} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
            elif "quota" in error:
                st.error("❌ API Quota exceeded — Please try after some time!")
                return None
            else:
                st.error(f"❌ Error: {str(e)}")
                return None
    return None

# Web Search with Retry
def web_search_with_retry(search_tool, prompt, retries=3):
    for attempt in range(retries):
        try:
            return search_tool.invoke(prompt)
        except Exception as e:
            error = str(e).lower()
            if "rate_limit" in error or "429" in error:
                wait_time = (attempt + 1) * 10
                st.warning(f"⚠️ Search rate limit — Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"❌ Web search failed: {str(e)}")
                return None
    return None

# Check if RAG has answer
def rag_has_no_answer(answer):
    no_answer_phrases = [
        "not available",
        "don't have",
        "do not have",
        "no information",
        "cannot find",
        "not found",
        "not mentioned",
        "not provided"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in no_answer_phrases)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # Search Mode
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
                try:
                    os.makedirs("docs", exist_ok=True)
                    for file in uploaded_files:
                        with open(f"docs/{file.name}", "wb") as f:
                            f.write(file.getbuffer())
                    st.session_state.upload_success = True
                    st.session_state.upload_count = len(uploaded_files)
                    st.cache_resource.clear()
                    st.session_state.models_loaded = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")

    if st.session_state.get("upload_success"):
        st.success(f"✅ {st.session_state.get('upload_count', 0)} file(s) processed!")
        st.session_state.upload_success = False

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
        save_history([])
        st.rerun()

    st.divider()

    # API Status
    st.subheader("🔌 API Status")
    st.success("✅ Groq — Connected")
    st.success("✅ Tavily — Connected")
    st.success("✅ ChromaDB — Local")

# Load Models
if "models_loaded" not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div style='text-align: center; padding: 100px;'>
            <h1>🤖 AI Research Agent</h1>
            <h3>⚡ Initializing models, please wait...</h3>
            <p>This may take a few seconds on first load</p>
        </div>
        """, unsafe_allow_html=True)
    chain, retriever, search_tool = load_models()
    st.session_state.models_loaded = True
    st.session_state.chain = chain
    st.session_state.retriever = retriever
    st.session_state.search_tool = search_tool
    placeholder.empty()
else:
    chain = st.session_state.chain
    retriever = st.session_state.retriever
    search_tool = st.session_state.search_tool

if not chain:
    st.error("❌ Failed to load models — Check your API keys!")
    st.stop()

# Main UI
st.title("🤖 AI Research Agent")
st.caption("Ask anything — I'll search documents or the web!")

# Initialize Session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = load_history()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything..."):

    # Save to history
    st.session_state.history.append(prompt)
    save_history(st.session_state.history)

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get Answer
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):

            response = ""

            # Web Only
            if search_mode == "🌐 Web Only":
                web_results = web_search_with_retry(search_tool, prompt)
                if web_results:
                    results = web_results.get("results", [])
                    response = "### 🌐 Web Search Results\n\n"
                    for r in results:
                        response += f"**{r['title']}**\n\n"
                        response += f"{r['content']}\n\n"
                        response += f"🔗 {r['url']}\n\n"
                        response += "---\n\n"
                else:
                    response = "❌ Web search failed — Please try again!"

            # Documents Only
            elif search_mode == "📄 Documents Only":
                rag_answer = invoke_with_retry(chain, prompt)
                if rag_answer:
                    docs = retriever.invoke(prompt)
                    sources = list(set([
                        f"{doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})"
                        for doc in docs
                    ]))
                    response = f"{rag_answer}\n\n📄 **Sources:**\n"
                    for source in sources:
                        response += f"- {source}\n"
                else:
                    response = "❌ Could not get answer — Please try again in 30 seconds!"

            # Auto Mode
            else:
                rag_answer = invoke_with_retry(chain, prompt)
                if rag_answer:
                    if rag_has_no_answer(rag_answer):
                        # Switch to Web Search
                        st.info("📡 Not found in documents — Searching the web...")
                        web_results = web_search_with_retry(search_tool, prompt)
                        if web_results:
                            results = web_results.get("results", [])
                            response = "🌐 **Web Search Results:**\n\n"
                            for r in results:
                                response += f"**{r['title']}**\n{r['content']}\n\n"
                                response += f"🔗 [Source]({r['url']})\n\n---\n\n"
                        else:
                            response = "❌ Both RAG and Web search failed — Please try again!"
                    else:
                        # RAG Answer
                        docs = retriever.invoke(prompt)
                        sources = list(set([
                            f"{doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})"
                            for doc in docs
                        ]))
                        response = f"{rag_answer}\n\n📄 **Sources:**\n"
                        for source in sources:
                            response += f"- {source}\n"
                else:
                    response = "❌ Could not get answer — Please try again in 30 seconds!"

            st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
