import os
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data", "weekly_topics.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Ensure the file exists.")
        
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            course_json = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {json_path}: {e}")

    docs = [Document(page_content=topics, metadata={"week": week}) for week, topics in course_json.items()]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 1})

retriever = get_retriever()