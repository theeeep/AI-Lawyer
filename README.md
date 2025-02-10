# AI Lawyer

AI Lawyer is an intelligent document analysis system that allows users to upload legal documents and interact with them through natural language queries. Built with modern AI technologies, it provides accurate and context-aware responses to legal questions.

## Features

- 📄 PDF Document Upload: Upload and process legal documents in PDF format
- 💬 Natural Language Queries: Ask questions about your documents in plain English
- 🔍 Intelligent Search: Uses RAG (Retrieval Augmented Generation) for accurate information retrieval
- 🧠 Advanced Language Model: Powered by Groq's deepseek-r1-distill-llama-70b model
- 🎯 Context-Aware Responses: Provides answers based on the specific content of your documents

## Prerequisites

- Python 3.13 or higher
- Environment variables set up in `.env` file

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/theeeep/AI-Lawyer.git
   cd ai-lawyer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install .
   ```

## Configuration

Create a `.env` file in the project root and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run src/frontend/frontend.py
   ```

2. Upload a PDF document using the file uploader
3. Enter your question in the text area
4. Click "Ask AI Lawyer" to get your response

## Project Structure

```
├── src/
│   ├── backend/
│   │   ├── rag_pipeline.py    # RAG implementation
│   │   └── vector_db.py       # Vector database operations
│   └── frontend/
│       └── frontend.py        # Streamlit UI
├── pdfs/                      # Storage for uploaded PDFs
├── vectorstore/               # FAISS vector database storage
└── docs/                      # Documentation
```

## Dependencies

- langchain - For RAG pipeline and LLM integration
- streamlit - Web interface
- faiss-cpu - Vector similarity search
- pdfplumber - PDF processing
- python-dotenv - Environment variable management


