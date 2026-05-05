from langchain_core.prompts import ChatPromptTemplate
import json
from state import GraphState, RouteDecision, FinalGeneration, EvaluationResult
from config import llm, ddg_search
from vector_store import retriever

def route_question(state: GraphState):
    print("NODE 1: ROUTING QUESTION")
    router_llm = llm.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent router. Decide if the user's question relates to a data science/AI course curriculum or if it is a general/math question."),
        ("human", "{question}")
    ])
    chain = prompt | router_llm
    decision = chain.invoke({"question": state["question"]})
    return {"route": decision.route}

def retrieve_and_search(state: GraphState):
    print("NODE 2: RETRIEVING FROM VECTOR STORE & WEB SEARCH")
    question = state["question"]
    
    retrieved_docs = retriever.invoke(question)
    best_match = retrieved_docs[0]
    week_num = best_match.metadata["week"]
    topics = best_match.page_content
    
    search_query = f"{topics} context: {question}"
    print(f"Executing web search for: {search_query}")
    try:
        search_results = ddg_search.invoke(search_query)
    except Exception:
        search_results = "Web search unavailable."

    return {
        "week_number": week_num,
        "topic": topics,
        "search_results": search_results
    }

def direct_answer(state: GraphState):
    print("NODE 3: DIRECT ANSWER (SKIPPING RETRIEVAL)")
    return {
        "week_number": "N/A",
        "topic": "General Knowledge / Off-topic",
        "search_results": "No search required."
    }

def generate_output(state: GraphState):
    print("NODE 4: GENERATING FINAL OUTPUT")
    generator_llm = llm.with_structured_output(FinalGeneration)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful teaching assistant. Answer the user's question based on the provided context.
        If the question is course-related, incorporate the week number and topics provided.
        Format your response exactly according to the required schema."""),
        ("human", """Question: {question}
        
        Retrieved Syllabus Week: {week}
        Syllabus Topics: {topic}
        Web Search Context: {search}
        
        Generate the final answer.""")
    ])
    
    chain = prompt | generator_llm
    result = chain.invoke({
        "question": state["question"],
        "week": state.get("week_number", "N/A"),
        "topic": state.get("topic", "N/A"),
        "search": state.get("search_results", "None")
    })
    
    return {"final_output": result.dict()}

def evaluate_output(state: GraphState):
    print("NODE 5: EVALUATING GENERATED OUTPUT")
    evaluator_llm = llm.with_structured_output(EvaluationResult)
    
    system = """You are a quality evaluated Evaluate the results against the original goal using this rubric:
    - Completeness: Does it fully address the goal? (0-0.4)
    - Accuracy: Is the information correct and specific? (0-0.3)
    - Clarity: Is it well-structured and clear? (0-0.3)
    Sum the scores for a total between 0.0 and 1.0
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Original Goal/Question: {question}
        Generated Output: {output}
        
        Evaluate the output and provide the scores and brief feedback.""")
    ])
    
    chain = prompt | evaluator_llm
    output_str = json.dumps(state["final_output"], indent=2)
    
    result = chain.invoke({
        "question": state["question"],
        "output": output_str
    })
    
    return {"evaluation": result.dict()}

def decide_next_node(state: GraphState):
    if state["route"] == "course_related":
        return "retrieve"
    return "direct_answer"