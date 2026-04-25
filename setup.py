import os

os.makedirs('rag', exist_ok=True)

# loader.py
with open('rag/loader.py', 'w', encoding='utf-8') as f:
    f.write("""from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(folder_path="docs/"):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    print(f"✅ Loaded {len(documents)} pages")
    return documents
""")
print("✅ loader.py created!")

# chunker.py
with open('rag/chunker.py', 'w', encoding='utf-8') as f:
    f.write("""from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks
""")
print("✅ chunker.py created!")

# embedder.py
with open('rag/embedder.py', 'w', encoding='utf-8') as f:
    f.write("""from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    print("✅ Gemini Embeddings model loaded")
    return embeddings
""")
print("✅ embedder.py created!")

# vectorstore.py
with open('rag/vectorstore.py', 'w', encoding='utf-8') as f:
    f.write("""from langchain_community.vectorstores import Chroma

def create_vectorstore(chunks, embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("✅ Vectorstore created & saved")
    return vectorstore

def load_vectorstore(embeddings):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    print("✅ Vectorstore loaded")
    return vectorstore
""")
print("✅ vectorstore.py created!")

# __init__.py
with open('rag/__init__.py', 'w', encoding='utf-8') as f:
    f.write("")
print("✅ __init__.py created!")

print("\n🎉 All files created successfully!")