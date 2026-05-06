from langchain_core.prompts import ChatPromptTemplate
import json
from pydantic import BaseModel, Field
from state import GraphState, LegacyState, RouteDecision, FinalGeneration, EvaluationResult

def legacy_route_question(state: LegacyState):
    print("LEGACY NODE 1: ROUTING QUESTION")
    router_llm = llm.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent router. Decide if the user's question relates to a data science/AI course curriculum or if it is a general/math question."),
        ("human", "{question}")
    ])
    chain = prompt | router_llm
    decision = chain.invoke({"question": state["question"]})
    return {"route": decision.route}

def legacy_generate_output(state: LegacyState):
    print("LEGACY NODE 4: GENERATING FINAL OUTPUT")
    generator_llm = llm.with_structured_output(FinalGeneration)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful teaching assistant. Answer the user's question based on the provided context.\nIf the question is course-related, incorporate the week number and topics provided.\nFormat your response exactly according to the required schema."),
        ("human", "Question: {question}\n\nRetrieved Syllabus Week: {week}\nSyllabus Topics: {topic}\nWeb Search Context: {search}\n\nGenerate the final answer.")
    ])
    
    chain = prompt | generator_llm
    result = chain.invoke({
        "question": state["question"],
        "week": state.get("week_number", "N/A"),
        "topic": state.get("topic", "N/A"),
        "search": state.get("search_results", "None")
    })
    
    return {"final_output": result.dict()}
from config import llm, ddg_search
from vector_store import retriever

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def route_question(state: GraphState):
    print("NODE 1: ROUTING QUESTION")
    router_llm = llm.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent router. Decide if the user's question relates to a data science/AI course curriculum or if it is a general/math question.\nChat History:\n{chat_history}"),
        ("human", "{question}")
    ])
    chain = prompt | router_llm
    history_str = "\n".join(state.get("chat_history", []))
    decision = chain.invoke({"question": state["question"], "chat_history": history_str})
    return {"route": decision.route}

def retrieve_and_search(state: GraphState):
    print("NODE 2: RETRIEVING FROM VECTOR STORE & WEB SEARCH")
    question = state["question"]
    
    retrieved_docs = retriever.invoke(question)
    best_match = retrieved_docs[0]
    
    grader_llm = llm.with_structured_output(GradeDocuments)
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved syllabus topic to a user question.\nIf the topic contains keyword(s) or semantic meaning related to the user question, grade it as 'yes'.\nOtherwise, grade it as 'no'."),
        ("human", "Retrieved topic: \n\n {topic} \n\n User question: {question}")
    ])
    
    grader_chain = grade_prompt | grader_llm
    relevance = grader_chain.invoke({"topic": best_match.page_content, "question": question})
    
    if relevance.binary_score.lower() == "yes":
        week_num = best_match.metadata["week"]
        topics = best_match.page_content
        search_query = f"{topics} context: {question}"
    else:
        week_num = "Out of Syllabus"
        topics = "N/A"
        search_query = question
        
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
        ("system", "You are a helpful teaching assistant. Answer the user's question based on the provided context.\nIf the question is course-related, incorporate the week number and topics provided.\nFormat your response exactly according to the required schema.\n\nLong-Term Memory:\n{long_term_memory}\n\nRecent Chat History:\n{chat_history}"),
        ("human", "Question: {question}\n\nRetrieved Syllabus Week: {week}\nSyllabus Topics: {topic}\nWeb Search Context: {search}\n\nGenerate the final answer.")
    ])
    
    chain = prompt | generator_llm
    
    history_str = "\n".join(state.get("chat_history", []))
    lt_memory = state.get("long_term_memory", "")
    
    result = chain.invoke({
        "question": state["question"],
        "week": state.get("week_number", "N/A"),
        "topic": state.get("topic", "N/A"),
        "search": state.get("search_results", "None"),
        "chat_history": history_str,
        "long_term_memory": lt_memory
    })
    
    current_history = state.get("chat_history", [])
    new_history = current_history + [f"AI: {result.text}"]
    
    return {
        "final_output": result.dict(),
        "chat_history": new_history
    }

def evaluate_output(state: GraphState):
    print("NODE 5: EVALUATING GENERATED OUTPUT")
    evaluator_llm = llm.with_structured_output(EvaluationResult)
    
    system = "You are a quality evaluated Evaluate the results against the original goal using this rubric:\n- Completeness: Does it fully address the goal? (0-0.4)\n- Accuracy: Is the information correct and specific? (0-0.3)\n- Clarity: Is it well-structured and clear? (0-0.3)\nSum the scores for a total between 0.0 and 1.0"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Original Goal/Question: {question}\nGenerated Output: {output}\n\nEvaluate the output and provide the scores and brief feedback.")
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