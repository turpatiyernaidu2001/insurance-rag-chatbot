from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

all_documents = []

folders = [
    "data/policy_docs",
    "data/claim_guides",
    "data/faq_docs"
]

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder, file)

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            all_documents.extend(documents)

print("Total Pages:", len(all_documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")

print(chunks[0].page_content[:1000])