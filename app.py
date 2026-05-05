import json
from graph import build_graph 

if __name__ == "__main__":
    app = build_graph()

    while(True):
        user_input = input("Ask a question or type 'exit' to quit: ")
        if user_input.lower() == "exit":
            break
        result = app.invoke({"question": user_input})
        
        output = result.get("final_output", {})
        print("\nFINAL JSON OUTPUT:")
        print(json.dumps(output, indent=2))
        
        if "text" in output:
            print(f"\nAnswer: {output['text']}")
        if "details" in output:
            print(f"Details: {output['details']}")
        if "week_number" in output and "topic" in output:
            print(f"From Week {output['week_number']}: {output['topic']}")
        print("-" * 60)
