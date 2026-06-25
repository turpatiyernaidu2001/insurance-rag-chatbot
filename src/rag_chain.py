from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
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

