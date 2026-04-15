# RAG-based PDF Question Answering System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions from PDF documents using semantic search and LLMs.

## Key Features
- Chunking (~500 characters with 50 overlap)
- Embedding generation for semantic similarity
- FAISS vector database for retrieval
- Top-k (3–5) chunk retrieval per query
- Context-grounded answer generation using LLM

## Tech Stack
- Python
- LangChain
- FAISS
- LLM APIs (OpenAI / HuggingFace)

## Workflow
1. Extract text from PDF
2. Split into chunks
3. Generate embeddings
4. Store in FAISS
5. Retrieve relevant chunks
6. Generate answer using LLM

## Example
User asks a question → system retrieves context → LLM generates answer

## Future Improvements
- Add Streamlit UI
- Improve retrieval ranking
- Multi-document support
