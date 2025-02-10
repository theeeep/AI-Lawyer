import os
import sys

import streamlit as st

from src.backend.rag_pipeline import answer_query, llm_model, retreive_docs

# Add the project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)


upload_file = st.file_uploader("Upload file", type="pdf", accept_multiple_files=False)

user_query = st.text_area(
    "Enter your prompt:", height=150, placeholder="Ask your Query"
)

ask_question = st.button("Ask AI Laywer")

if upload_file:
    # Save the uploaded file
    from src.backend.vector_db import process_uploaded_document, upload_pdf

    upload_pdf(upload_file)

    # Process the uploaded document
    file_path = f"pdfs/{upload_file.name}"
    with st.spinner("Processing document..."):
        process_uploaded_document(file_path)
    st.success("Document processed successfully!")

if ask_question:
    if upload_file:
        st.chat_message("user").write(user_query)

        # RAG Pipeline call
        retrieved_docs = retreive_docs(user_query)

        response = answer_query(
            documents=retrieved_docs, model=llm_model, query=user_query
        )

        st.chat_message("ai").write(response)
    else:
        st.error("Please upload a file")
