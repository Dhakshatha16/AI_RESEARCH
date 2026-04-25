from langchain_community.vectorstores import Chroma

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
