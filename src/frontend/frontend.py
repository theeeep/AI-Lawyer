import streamlit as st

from src.backend.rag_pipeline import answer_query, llm_model, retreive_docs

upload_file = st.file_uploader("Upload file", type="pdf", accept_multiple_files=False)

user_query = st.text_area(
    "Enter your prompt:", height=150, placeholder="Ask your Query"
)

ask_question = st.button("Ask AI Laywer")

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
