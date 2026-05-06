# Agentic Syllabus Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜🔗_LangChain-1.2.13-orange.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.3-blueviolet.svg)](https://python.langchain.com/docs/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.23-green.svg)](https://www.trychroma.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.13.3-red.svg)](https://docs.pydantic.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-1.13.2-blue.svg)](https://github.com/facebookresearch/faiss)
[![DuckDuckGo](https://img.shields.io/badge/DuckDuckGo_Search-8.1.1-lightgrey.svg)](https://github.com/deedy5/duckduckgo_search)
[![Groq](https://img.shields.io/badge/Powered_by-Groq-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

The **Agentic Syllabus Assistant** is an intelligent, multi-agent AI system built using LangGraph and LangChain. It acts as an automated teaching assistant designed to answer questions related to a data science and machine learning curriculum. 

Instead of a standard flat prompt, this project utilizes an **Agentic Workflow**:
1. It intelligently routes user questions based on their intent.
2. It retrieves highly specific curriculum data (syllabus weeks and topics) from a local FAISS vector store.
3. It performs live web searches via DuckDuckGo to augment its knowledge.
4. It synthesizes this information using a Groq-powered LLM (`llama-3.3-70b-versatile`) to provide structured, context-aware answers.
5. It features an automated **Quality Evaluator** node that scores every response based on completeness, accuracy, and clarity.

## Key Features

*   **Modular Architecture**: Separated concerns across config, state, nodes, and graph modules for better maintainability and testing.
*   **Structured Output**: Uses Pydantic models (`RouteDecision`, `FinalGeneration`, `EvaluationResult`) to ensure reliable and consistent LLM responses.
*   **Self-Correcting Retrieval**: Node-level relevance grading ensures that retrieved syllabus content is actually related to the query before proceeding to web search.
*   **Dual-Graph Strategy**: Supports both a production graph (for `app.py`) and an evaluation graph (for `test.py`) which includes the quality evaluator node.
*   **Integrated Performance Monitoring**: Custom callback handlers track token usage and latency in real-time.

## Realistic Use Cases

1. **Persistent Tutoring Across Sessions (Long-Term Memory):** 
   A student asks complex questions about Artificial Neural Networks and ends their session. A few days later, they return using the same `thread_id` and ask, "Can you explain the math behind the backpropagation for those networks we discussed?" The system leverages **ChromaDB** to recall the previous session's context and provides a seamless, context-aware answer without requiring the student to repeat themselves.

2. **Automated Study Notes Generation:**
   During intensive exam preparation, a student engages in a multi-turn deep dive on a specific week's topics. The system tracks the conversation utilizing its short-term **InMemorySaver** checkpointing. Once the conversation reaches 4 turns, the assistant automatically exports a beautifully formatted Markdown transcript to the `data/transcript` folder. The student can then use this `.md` file as a pre-made, highly accurate study guide.

## Visual Walkthrough

[![Agentic Syllabus Assistant](https://mermaid.ink/img/pako:eNplkt9umzAUxl_F8sXUSiELIQl_pG7qylZFajqpsIsN0GSF04AWbHQwbbIoL7EX2JvsmfYIM8bJ6MaF7WN9v8_HnznQtciBBnSDrC5IHKacqO9TA3iRdCNZ8rqV2SWxrDfkRvDHcpPEBQLLyTIkr81W1mN9oaV38errLcjkpkBRsfBdQB5AYglPQO4E31gxYNXpJeykoQ2i8cjgK6gE7iP2BBgoUJ0aFQJlj0eSSTBwP0YDiwfRSsBDP5F7dc3jUGr2LUt10WIDqr-tsst71vSanJt-RSJguC60UfavxS1wQLbVbFgirGXST-SaN8_m-Bedno07pMclJKcF-dhKFfsQM35D-dDvTOr0VF4qzaSbSdwiJ1KQ00sYP6Mx79ryList1QW5uiKzty8CM6Luup-h0dz7Xa0eI_n988cvsyarkMTIeLPGspbZf-S9MPnW2_1F8qHkKrTrpaqbWvAGssueMGZnacrpSP2iZU4DiS2MaKXen3UlPXRESmUBFaQ0UMuc4beUpvyomJrxL0JUJwxFuylo8Mi2jaraOleJhSVTP_9fCfAcULdMA8edaw8aHOiOBra_GM-ntu8tZp7r-a4zonsaWLY3GS_cqbewPXsy96a2cxzR7_rYydj3FxPXd6Zzz3Pmzsw__gEv5xbO?type=png)](https://mermaid.live/edit#pako:eNplkt9umzAUxl_F8sXUSiELIQl_pG7qylZFajqpsIsN0GSF04AWbHQwbbIoL7EX2JvsmfYIM8bJ6MaF7WN9v8_HnznQtciBBnSDrC5IHKacqO9TA3iRdCNZ8rqV2SWxrDfkRvDHcpPEBQLLyTIkr81W1mN9oaV38errLcjkpkBRsfBdQB5AYglPQO4E31gxYNXpJeykoQ2i8cjgK6gE7iP2BBgoUJ0aFQJlj0eSSTBwP0YDiwfRSsBDP5F7dc3jUGr2LUt10WIDqr-tsst71vSanJt-RSJguC60UfavxS1wQLbVbFgirGXST-SaN8_m-Bedno07pMclJKcF-dhKFfsQM35D-dDvTOr0VF4qzaSbSdwiJ1KQ00sYP6Mx79ryList1QW5uiKzty8CM6Luup-h0dz7Xa0eI_n988cvsyarkMTIeLPGspbZf-S9MPnW2_1F8qHkKrTrpaqbWvAGssueMGZnacrpSP2iZU4DiS2MaKXen3UlPXRESmUBFaQ0UMuc4beUpvyomJrxL0JUJwxFuylo8Mi2jaraOleJhSVTP_9fCfAcULdMA8edaw8aHOiOBra_GM-ntu8tZp7r-a4zonsaWLY3GS_cqbewPXsy96a2cxzR7_rYydj3FxPXd6Zzz3Pmzsw__gEv5xbO)

## Project Structure 

*   **`app.py`**: Interactive CLI for chatting with the assistant.
*   **`main.py`**: Standard execution script for testing specific queries.
*   **`test.py`**: Comprehensive batch testing and analytics suite.
*   **`graph.py`**: LangGraph construction logic (Production and Evaluation graphs).
*   **`nodes.py`**: Implementation of all individual agent nodes (Router, Retriever, Generator, Evaluator).
*   **`state.py`**: Pydantic and TypedDict definitions for graph state and structured output schemas.
*   **`vector_store.py`**: FAISS initialization and syllabus retrieval logic.
*   **`config.py`**: Centralized configuration for LLMs, tools, and environment variables.

## Setup & Installation

### Prerequisites
* Python 3.10 or higher.
* A free API key from [Groq Console](https://console.groq.com/).

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the Assistant

### 1. Interactive Mode
Launch the interactive CLI to ask questions in real-time:
```bash
python app.py
```

### 2. Batch Evaluation & Analytics
Run the automated batch test to see the system's performance metrics (Latency, Token Usage, AI Scores):
```bash
python test.py
```

## How It Works: The Nodes Explained

The system is powered by a **State Graph** (LangGraph) consisting of five distinct nodes:

### 1. Node 1: Router (`route_question`)
The entry point. It analyzes the user's intent using structured output to decide whether to search the syllabus or answer directly.

### 2. Node 2: Retrieve & Search (`retrieve_and_search`)
Performs a hybrid search using a **FAISS Vector Store** (syllabus) and **DuckDuckGo** (live web context). Includes an internal **Relevance Grader** to validate retrieved documents.

### 3. Node 3: Direct Answer (`direct_answer`)
A bypass node for general knowledge or math questions, saving API tokens and reducing latency.

### 4. Node 4: Generate Output (`generate_output`)
Synthesizes all gathered information into a structured JSON response (via Pydantic) to ensure high-quality formatting.

### 5. Node 5: Quality Evaluator (`evaluate_output`)
An automated critic (only in the evaluation graph) that scores the response on:
- **Completeness (0.4)**
- **Accuracy (0.3)**
- **Clarity (0.3)**
- **Total Score (1.0)**

## Performance Tracking

The `test.py` script provides detailed analytics including:
- **Average Latency**: Response time per question across the batch.
- **Token Consumption**: Real-time tracking of prompt and completion tokens via a custom callback handler.
- **AI-Driven Scores**: Automated quality metrics generated by the Evaluator node.
- **Route Tracking**: Percentage of questions handled by the syllabus retriever vs. direct answers.