import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


from langchain_community.vectorstores import FAISS
from src.backend.vector_db import get_embedding_model, FAISS_DB_PATH, ollama_model

dotenv.load_dotenv()

llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")


# Load vector database
def load_vector_db():
    try:
        return FAISS.load_local(
            FAISS_DB_PATH,
            get_embedding_model(ollama_model),
            allow_dangerous_deserialization=True  # Safe since we create and load the database locally
        )
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

# Retrieve Docs
def retreive_docs(query):
    faiss_db = load_vector_db()
    if faiss_db is None:
        print("No vector database found. Please upload a document first.")
        return []
    
    try:
        # Get relevant documents with similarity search
        docs = faiss_db.similarity_search(query, k=4)  # Retrieve top 4 most relevant chunks
        if not docs:
            print(f"No relevant documents found for query: {query}")
        return docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


def get_context(documents):
    if not documents:
        return "No relevant documents found. Please ensure a document has been uploaded."
    
    # Join document contents with clear separation and metadata
    contexts = []
    for doc in documents:
        # Add page content with metadata if available
        context = f"Content: {doc.page_content}"
        if hasattr(doc.metadata, 'page') and doc.metadata['page']:
            context = f"[Page {doc.metadata['page']}] {context}"
        contexts.append(context)
    
    return "\n\n---\n\n".join(contexts)


custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    
    # Debug print to check context
    print("\nRetrieved Context:")
    print("=================")
    print(context)
    print("=================\n")
    
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    return chain.invoke({"question": query, "context": context})
