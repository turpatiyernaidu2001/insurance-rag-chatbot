from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

while True:
    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer based only on the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

docs = retriever.invoke(query)

for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:500])