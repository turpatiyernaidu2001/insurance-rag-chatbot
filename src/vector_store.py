import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load API Key
load_dotenv()

print("API Key Loaded:", bool(os.getenv("GOOGLE_API_KEY")))
print("API Key Value:", os.getenv("GOOGLE_API_KEY")[:10] if os.getenv("GOOGLE_API_KEY") else "NOT FOUND")

all_documents = []

folders = [
    "data/policy_docs",
    "data/claim_guides",
    "data/faq_docs"
]

# Load PDFs
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            all_documents.extend(documents)

print("Total Pages:", len(all_documents))

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)

print("Total Chunks:", len(chunks))


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS Vector Store
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save FAISS
vectorstore.save_local("faiss_index")

print("FAISS Index Created Successfully!")