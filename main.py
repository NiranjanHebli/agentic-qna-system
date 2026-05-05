import json
from graph import build_graph 

if __name__ == "__main__":
    app = build_graph()

    print("\n\n=== EXAMPLE 1: Course Related Question ===")
    example_1_inputs = {"question": "Can you explain how Self-Attention works?"}
    result_1 = app.invoke(example_1_inputs)
    print("\nFINAL JSON OUTPUT:")
    print(json.dumps(result_1["final_output"], indent=2))

    print("\n\n=== EXAMPLE 2: General / Math Question ===")
    example_2_inputs = {"question": "What is the square root of 144?"}
    result_2 = app.invoke(example_2_inputs)
    print("\nFINAL JSON OUTPUT:")
    print(json.dumps(result_2["final_output"], indent=2))
