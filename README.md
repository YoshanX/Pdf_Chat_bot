# Pdf_Chat_bot

📄 PDF Question Answering Bot (RAG + LangChain + Streamlit)

This project implements a Retrieval-Augmented Generation (RAG) architecture using LangChain, where uploaded PDFs are converted into searchable knowledge bases. The app extracts text, splits it into chunks, embeds them with MiniLM, stores them in a FAISS vector database, retrieves the most relevant content, and uses a Mistral LLM to generate accurate answers.

🚀 Features

📂 Upload any PDF and automatically build a knowledge base

🔍 FAISS vector search with MiniLM embeddings

🤖 RAG-based question answering using Mistral-7B

💬 Interactive chat interface powered by Streamlit

⚡ Fast and efficient local embedding generation

♻️ Chat history support and clear-reset option

🔐 Easy .env-based API key setup

🛠️ Technologies Used

Python, Streamlit

LangChain, FAISS

HuggingFace (Inference API)

MiniLM embeddings

Mistral-7B-Instruct
