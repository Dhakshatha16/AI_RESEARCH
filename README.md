
# AI Research Agent (RAG-Based)

## Overview

AI Research Agent is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

The system retrieves relevant document chunks using vector search and generates answers using a Large Language Model (LLM).

## Features

* PDF document upload
* Automatic document chunking
* Semantic search using embeddings
* ChromaDB vector database
* Source citation support
* Streamlit web interface
* AWS EC2 deployment
* Groq LLM integration

## Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Groq (Llama 3.3 70B)
* Streamlit
* AWS EC2
* Git & GitHub

## Project Structure

AI_RESEARCH/

├── app.py

├── rag/

│   ├── embedder.py

│   ├── loader.py

│   ├── chunker.py

│   ├── retriever.py

│   └── vectorstore.py

├── docs/

├── chroma_db/

├── requirements.txt

└── README.md

## Deployment

The application is deployed on AWS EC2 and can be accessed through a public endpoint.

## How It Works

1. Upload PDF documents
2. Extract text from PDFs
3. Split documents into chunks
4. Generate embeddings
5. Store embeddings in ChromaDB
6. Retrieve relevant chunks for user queries
7. Generate answers using Groq LLM
8. Display sources along with responses

## Future Improvements

* Docker deployment
* CI/CD with GitHub Actions
* HTTPS and custom domain
* User authentication
* Multi-document knowledge base support
# AI Research Agent (RAG-Based)

## Overview

AI Research Agent is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

The system retrieves relevant document chunks using vector search and generates answers using a Large Language Model (LLM).

## Features

* PDF document upload
* Automatic document chunking
* Semantic search using embeddings
* ChromaDB vector database
* Source citation support
* Streamlit web interface
* AWS EC2 deployment
* Groq LLM integration

## Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Groq (Llama 3.3 70B)
* Streamlit
* AWS EC2
* Git & GitHub

## Project Structure

AI_RESEARCH/

├── app.py

├── rag/

│   ├── embedder.py

│   ├── loader.py

│   ├── chunker.py

│   ├── retriever.py

│   └── vectorstore.py

├── docs/

├── chroma_db/

├── requirements.txt

└── README.md

## Deployment

The application is deployed on AWS EC2 and can be accessed through a public endpoint.

## How It Works

1. Upload PDF documents
2. Extract text from PDFs
3. Split documents into chunks
4. Generate embeddings
5. Store embeddings in ChromaDB
6. Retrieve relevant chunks for user queries
7. Generate answers using Groq LLM
8. Display sources along with responses

## Future Improvements

* Docker deployment
* CI/CD with GitHub Actions
* HTTPS and custom domain
* User authentication
* Multi-document knowledge base support
