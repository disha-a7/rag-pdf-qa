
# RAG-based PDF Question Answering System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions from PDF documents using semantic search and LLMs.

## Features
- Text extraction and chunking (~500 characters with 50 overlap)
- Embedding generation for semantic similarity
- FAISS vector database for retrieval
- Top-k (3–5) chunk retrieval per query
- Context-grounded answer generation using LLM

## Tech Stack
- Python
- LangChain
- FAISS
- LLM APIs (OpenAI/HuggingFace)

## How It Works
1. Load and extract text from PDF
2. Split text into chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS
5. Retrieve relevant chunks for a query
6. Generate answer using LLM

## Example Use Case
User asks a question → system retrieves relevant context → LLM generates answer

## Future Improvements
- Add UI (Streamlit)
- Improve retrieval ranking
- Support multiple documents
