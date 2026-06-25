from langchain_community.document_loaders import PyPDFLoader
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

            print(f"Loading: {pdf_path}")

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            all_documents.extend(documents)

print("\nTotal Pages Loaded:", len(all_documents))

print("\nSample Content:\n")

print(all_documents[0].page_content[:500])