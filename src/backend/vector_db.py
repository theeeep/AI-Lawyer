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


file_path = "universal_declaration_of_human_rights.pdf"
documents = load_pdf(file_path)


# Create Chunks
def create_chinks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


text_chunks = create_chinks(documents)

# Embedding Models
ollama_model = "deepseek-r1:1.5b"


# Get Embedding Model
def get_embedding_model(ollama_model):
    embeddings = OllamaEmbeddings(model=ollama_model)
    return embeddings


# Index Documents and store in VectorDB (FAISS)
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model))
faiss_db.save_local(FAISS_DB_PATH)
