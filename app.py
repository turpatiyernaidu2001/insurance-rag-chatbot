from dotenv import load_dotenv
import streamlit as st

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

st.title("Insurance RAG Chatbot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat Input
question = st.chat_input("Ask insurance question")

if question:

    # Show User Message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    docs = retriever.invoke(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer based only on the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    answer = response.content

    # Save Assistant Message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Show Retrieved Chunks
    with st.expander("Retrieved Context"):
        for i, doc in enumerate(docs):
            st.write(f"Chunk {i+1}")
            st.write(doc.page_content)
            st.divider()

    st.rerun()