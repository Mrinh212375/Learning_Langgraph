import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv("key.env")
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")
response = llm.invoke("Say hello!")
print(response.content)