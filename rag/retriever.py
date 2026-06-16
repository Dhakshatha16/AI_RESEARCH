from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

def create_qa_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
You are a helpful AI Research Assistant.
Use the following context to answer the question accurately.
If the context contains relevant information, use it to give a detailed answer.
If the context does NOT contain relevant information about the question, respond with exactly:
"This information is not available in the uploaded documents."

Context: {context}
Question: {question}

Answer:""")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ QA Chain created!")
    return chain, retriever