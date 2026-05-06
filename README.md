# Agentic Syllabus Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜🔗_LangChain-latest-orange.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-blueviolet.svg)](https://python.langchain.com/docs/langgraph)
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

## Visual Walkthrough

[![HLD](https://mermaid.ink/img/pako:eNp1UGFvmzAU_CvWkzoRjUZgoMk8aVI2smpTl0oln1pPkwUOoIIdGbOui_LfZ8DQMm1IFva9u3vv3QlSmXEgcKjkU1owpdE-pgKZ7-ICxfxQCo52htIMYLLf3O0dJ9GGuVgM2J1sNVcnCh0P-cQClArn5uYbynhaGj06Ml0sKJytiGtV8p_8wcowmSD0BiWcqbToHD5vviQJeoviNn3szrU0Ht8Hj7hUPNWjQ0AsgDaieRr6f3w-sqZBanBm1Yv2mguumJ76h2SC0G2rj63uu0tVM41MCqxCX5Pb3YvBdhc7zlZkXQp_B7bN8llg6PLyw5jKq8gMjCj0d4JS2aqG_1C8MiNkFAaNjeS_qryfubL0Yf-BPI-5L48Lvk7vH4V5QD3BLDtumVZdonYU1f_ezyq2o2v9WapLKWaUyTq3F1MFF3JVZkC0arkLNTfBd084dUoKuuA1p0DMNWPqkQIVZ6M5MnEvZT3KzDx5AeTAqsa82mNmzOOS5YrVE6q4yLj6JFuhgay83gPICX4BwVG0DNfvVhHGYbQO_DBy4RmIH-Bl5OEwCHCAff8Kn1343Xf1llcRDn0v8lbYw364Ds9_AMtYCVg?type=png)](https://mermaid.live/edit#pako:eNp1UGFvmzAU_CvWkzoRjUZgoMk8aVI2smpTl0oln1pPkwUOoIIdGbOui_LfZ8DQMm1IFva9u3vv3QlSmXEgcKjkU1owpdE-pgKZ7-ICxfxQCo52htIMYLLf3O0dJ9GGuVgM2J1sNVcnCh0P-cQClArn5uYbynhaGj06Ml0sKJytiGtV8p_8wcowmSD0BiWcqbToHD5vviQJeoviNn3szrU0Ht8Hj7hUPNWjQ0AsgDaieRr6f3w-sqZBanBm1Yv2mguumJ76h2SC0G2rj63uu0tVM41MCqxCX5Pb3YvBdhc7zlZkXQp_B7bN8llg6PLyw5jKq8gMjCj0d4JS2aqG_1C8MiNkFAaNjeS_qryfubL0Yf-BPI-5L48Lvk7vH4V5QD3BLDtumVZdonYU1f_ezyq2o2v9WapLKWaUyTq3F1MFF3JVZkC0arkLNTfBd084dUoKuuA1p0DMNWPqkQIVZ6M5MnEvZT3KzDx5AeTAqsa82mNmzOOS5YrVE6q4yLj6JFuhgay83gPICX4BwVG0DNfvVhHGYbQO_DBy4RmIH-Bl5OEwCHCAff8Kn1343Xf1llcRDn0v8lbYw364Ds9_AMtYCVg)

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