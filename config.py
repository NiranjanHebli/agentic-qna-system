import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please make sure your .env file is set up correctly.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    api_key=groq_api_key
)

ddg_search = DuckDuckGoSearchRun()