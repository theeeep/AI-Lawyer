from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Upload and load Raw PDFs

pdfs_dir = "pdfs/"


# Upload file
def upload_pdf(file):
    with open(pdfs_dir + file.name, "wb") as f:
        f.write(file.getbuffer())


# Load file
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


# Process uploaded document
def process_uploaded_document(file_path):
    # Load the PDF
    documents = load_pdf(file_path)

    # Create chunks from the documents
    text_chunks = create_chunks(documents)

    # Create and save vector database
    embeddings = get_embedding_model(ollama_model)
    faiss_db = FAISS.from_documents(text_chunks, embeddings)

    # Save with proper serialization settings
    faiss_db.save_local(FAISS_DB_PATH)
    return faiss_db


# Create Chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# Embedding Models
ollama_model = "deepseek-r1:1.5b"


# Get Embedding Model
def get_embedding_model(ollama_model):
    embeddings = OllamaEmbeddings(model=ollama_model)
    return embeddings


# Path for storing the vector database
FAISS_DB_PATH = "vectorstore/db_faiss"
