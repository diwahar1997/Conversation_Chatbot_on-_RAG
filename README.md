# Conversational Chatbot with Retrieval-Augmented Generation (RAG)

This repository contains a **Conversational Chatbot** built using **Retrieval-Augmented Generation (RAG)** architecture. The chatbot is powered by a combination of vector databases for document retrieval (using FAISS) and language models (using Langchain and HuggingFace). It stores and manages chat sessions, supports retrieval-based responses, and allows history-aware conversations.

## Features

- **Real-time conversation**: The chatbot retrieves relevant documents and generates context-aware responses.
- **Session management**: Chat history is stored in MongoDB, enabling persistent conversations across sessions.
- **History-aware retriever**: Uses past interactions to generate more meaningful and contextual responses.
- **Asynchronous processing**: Chat history is stored asynchronously using a thread pool for efficient processing.
- **Modular design**: The code is modular and easy to extend for custom prompts and embeddings.

## Technologies Used

- **Python 3.8+**
- **Flask**: Web framework for serving API endpoints.
- **Langchain**: For managing the generative model and retrieval pipeline.
- **FAISS**: Vector search engine for similarity searches in the chatbot's knowledge base.
- **MongoDB**: NoSQL database for storing chatbot session data.
- **HuggingFace Transformers**: Pre-trained language models for generating responses.

## Prerequisites

- **Python 3.8+**
- **MongoDB**: You should have MongoDB installed and running.
- **FAISS**: Install FAISS for efficient vector search.
- **HuggingFace Transformers**: Use pre-trained models from HuggingFace.
- **Langchain**: Install Langchain for managing the conversational pipeline.
- **HashiCorp Vault**: For secure key management (optional, but recommended).

## Installation

1. **Clone the Repository**
   ```bash
  git clone https://github.com/your-username/conversation-chatbot-rag.git
  cd conversation-chatbot-rag
  python chatbot.py
