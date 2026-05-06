from typing import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field

class GraphState(TypedDict):
    question: str
    long_term_memory: str
    chat_history: list
    route: str
    week_number: str
    topic: str
    search_results: str
    final_output: dict
    evaluation: dict

class LegacyState(TypedDict):
    question: str
    route: str
    week_number: str
    topic: str
    search_results: str
    final_output: dict
    evaluation: dict

class RouteDecision(BaseModel):
    route: str = Field(description="Choose 'course_related' if the question is about AI, ML, programming, or the course curriculum. Choose 'general' for general knowledge, math, or off-topic questions.")

class FinalGeneration(BaseModel):
    text: str = Field(description="The direct answer to the user's question.")
    details: str = Field(description="Additional context, explanations, or web search summaries.")
    week_number: str = Field(description="The specific week number related to the query, or 'N/A'.")
    topic: str = Field(description="The extracted syllabus topics, or 'General Knowledge'.")

class EvaluationResult(BaseModel):
    completeness: float = Field(description="Completeness score between 0.0 and 0.4")
    accuracy: float = Field(description="Accuracy score between 0.0 and 0.3")
    clarity: float = Field(description="Clarity score between 0.0 and 0.3")
    total_score: float = Field(description="Sum of the scores (0.0 to 1.0)")
    feedback: str = Field(description="Brief explanation justifying the scores given.")