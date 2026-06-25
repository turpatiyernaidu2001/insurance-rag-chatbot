from dotenv import load_dotenv
load_dotenv()

from google import genai
import os

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents="What is insurance?"
)

print("Embedding created successfully!")
print("Vector length:", len(response.embeddings[0].values))
print(response.embeddings[0].values[:10])