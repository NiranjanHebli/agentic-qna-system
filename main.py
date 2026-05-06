import os
import json
import uuid
import datetime
import chromadb
from chromadb.config import Settings
from graph import build_graph

# Setup ChromaDB for Long-Term Memory
db_path = os.path.join(os.path.dirname(__file__), "data", "chroma_memory")
chroma_client = chromadb.PersistentClient(
    path=db_path, 
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection("chat_history_v2")

def save_to_chroma(session_id, user_msg, ai_msg):
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        documents=[f"User: {user_msg}\nAI: {ai_msg}"],
        metadatas=[{"session_id": session_id, "timestamp": datetime.datetime.now().isoformat()}]
    )

def get_past_context(query_text, session_id):
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=2,
            where={"session_id": session_id}
        )
        if results["documents"] and results["documents"][0]:
            return "\n".join(results["documents"][0])
    except Exception as e:
        pass
    return ""

def main():
    # Build graph with MemorySaver (Short-Term Memory)
    app = build_graph()
    
    print("\n" + "="*60)
    print(" Agentic Q&A System with Persistent Memory")
    print("="*60)
    
    # Prompt for thread id
    thread_id = input("Enter session/thread ID (or press enter for a new one): ").strip()
    if not thread_id:
        thread_id = str(uuid.uuid4())[:8]
        
    # LangGraph Short-Term Memory Checkpointer config
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n--- Starting session: {thread_id} ---")
    print("Type 'exit' to quit.\n")
    
    transcript = []
    turn_count = 0
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        # Retrieve Long-Term Memory
        past_context = get_past_context(user_input, thread_id)
        if past_context:
            print(f"[Long-Term Memory Retrieved: {len(past_context)} chars]")
        
        current_state = app.get_state(config).values
        current_history = current_state.get("chat_history", [])
        new_history = current_history + [f"User: {user_input}"]
        
        # Invoke Graph (Short-Term Memory handles state checkpointing)
        result = app.invoke({
            "question": user_input,
            "chat_history": new_history,
            "long_term_memory": past_context or ""
        }, config=config)
        
        output = result.get("final_output", {})
        
        ai_response = output.get("text", "No text provided.")
        if "details" in output:
            ai_response += f"\nDetails: {output['details']}"
            
        print(f"\nAI: {ai_response}")
        print("-" * 60)
        
        # Save to Long-Term Memory
        save_to_chroma(thread_id, user_input, ai_response)
        
        # Save to Transcript buffer
        transcript.append(f"**You:** {user_input}\n\n**AI:** {ai_response}\n\n---\n")
        turn_count += 1
        
        # Save 4-turn transcript
        if turn_count == 4:
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_dir = os.path.join(os.path.dirname(__file__), "data", "transcript")
            os.makedirs(transcript_dir, exist_ok=True)
            
            file_name = f"transcript_{thread_id}_{date_str}.md"
            file_path = os.path.join(transcript_dir, file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Transcript for Session {thread_id}\n\n")
                f.write("\n".join(transcript))
                
            print(f"\n[ Saved 4-turn transcript to {file_path}]\n")
            turn_count = 0
            transcript = [] 

if __name__ == "__main__":
    main()
