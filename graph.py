from langgraph.graph import StateGraph, END
from state import GraphState, LegacyState
from nodes import route_question, retrieve_and_search, direct_answer, generate_output, evaluate_output, decide_next_node, legacy_route_question, legacy_generate_output
from langgraph.checkpoint.memory import MemorySaver

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("router", route_question)
    workflow.add_node("retrieve", retrieve_and_search)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("generate", generate_output)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("direct_answer", "generate")
    workflow.add_edge("generate", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def build_test_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("router", route_question)
    workflow.add_node("retrieve", retrieve_and_search)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("generate", generate_output)
    workflow.add_node("evaluate", evaluate_output)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("direct_answer", "generate")
    
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def build_legacy_test_graph():
    workflow = StateGraph(LegacyState)

    workflow.add_node("router", legacy_route_question)
    workflow.add_node("retrieve", retrieve_and_search)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("generate", legacy_generate_output)
    workflow.add_node("evaluate", evaluate_output)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer"
        }
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("direct_answer", "generate")
    
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()