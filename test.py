import time
import json
from graph import build_legacy_test_graph
from langchain_core.callbacks import BaseCallbackHandler

class GroqTokenTracker(BaseCallbackHandler):

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        try:
            msg = response.generations[0][0].message

            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                self.prompt_tokens += msg.usage_metadata.get("input_tokens", 0)
                self.completion_tokens += msg.usage_metadata.get("output_tokens", 0)
                self.total_tokens += msg.usage_metadata.get("total_tokens", 0)

            elif (
                hasattr(msg, "response_metadata")
                and "token_usage" in msg.response_metadata
            ):
                usage = msg.response_metadata["token_usage"]
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)
        except Exception:
            pass

if __name__ == "__main__":
    print(" RUNNING MODULAR EVALUATION BATCH TEST ")

    app = build_legacy_test_graph()
    token_tracker = GroqTokenTracker()

    math_questions = [
        "What is the square root of 256?",
        "Solve the algebraic equation: 3x + 7 = 22 for x.",
        "What is the formula for the area of a circle, and what is the area if the radius is 5?",
        "What is the derivative of f(x) = 4x^3 - 2x + 1?",
        "If a car travels at a constant speed of 60 mph, how long will it take to travel 150 miles?",
    ]

    course_questions = [
        "What machine learning algorithms are covered in Week 6?",
        "Can you explain how TF-IDF works for text classification and which week covers it?",
        "What are the main differences between CNNs and RNNs as taught in the curriculum?",
        "Which week introduces Generative Adversarial Networks (GANs) and Foundation Models?",
        "Explain the concept of Retrieval-Augmented Generation (RAG) and mention the relevant week.",
    ]

    all_test_questions = math_questions + course_questions

    report_data = []
    start_batch_time = time.time()

    for i, question in enumerate(all_test_questions, 1):
        print(f"\n[Processing {i}/{len(all_test_questions)}] {question}")

        q_start_time = time.time()

        result = app.invoke(
            {"question": question}, 
            config={"callbacks": [token_tracker]}
        )

        q_latency = time.time() - q_start_time
        eval_scores = result.get("evaluation", {})

        report_data.append(
            {
                "id": i,
                "route": result.get("route", "Unknown"),
                "latency": q_latency,
                "completeness": eval_scores.get("completeness", 0),
                "accuracy": eval_scores.get("accuracy", 0),
                "clarity": eval_scores.get("clarity", 0),
                "total_score": eval_scores.get("total_score", 0),
            }
        )

    total_batch_time = time.time() - start_batch_time

    avg_latency = sum(r["latency"] for r in report_data) / len(report_data)
    avg_score = sum(r["total_score"] for r in report_data) / len(report_data)
    avg_comp = sum(r["completeness"] for r in report_data) / len(report_data)
    avg_acc = sum(r["accuracy"] for r in report_data) / len(report_data)
    avg_clarity = sum(r["clarity"] for r in report_data) / len(report_data)

    print(" \n              FINAL EVALUATION REPORT                 ")
    print("-" * 60)

    print("\n PERFORMANCE METRICS ")
    print(f"Total Execution Time : {total_batch_time:.2f} seconds")
    print(f"Average Latency / Q  : {avg_latency:.2f} seconds")
    print(f"Total Questions      : {len(all_test_questions)}")

    print("\n TOKEN CONSUMPTION (GROQ) ")
    print(f"Prompt Tokens        : {token_tracker.prompt_tokens:,}")
    print(f"Completion Tokens    : {token_tracker.completion_tokens:,}")
    print(f"Total Tokens         : {token_tracker.total_tokens:,}")

    print("\n AI EVALUATOR SCORES (Averages) ")
    print(f"Average Completeness : {avg_comp:.2f} / 0.40")
    print(f"Average Accuracy     : {avg_acc:.2f} / 0.30")
    print(f"Average Clarity      : {avg_clarity:.2f} / 0.30")
    print(f"Overall Average Score: {avg_score:.2f} / 1.00")

    print("\n QUESTION BREAKDOWN ")
    print(f"{'Q#':<4} | {'Route Taken':<15} | {'Latency(s)':<10} | {'Score':<5}")
    print("-" * 45)
    for r in report_data:
        print(
            f"{r['id']:<4} | {r['route']:<15} | {r['latency']:<10.2f} | {r['total_score']:<5.2f}"
        )
    print("=" * 45 + "\n")
