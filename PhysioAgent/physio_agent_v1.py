"""
Multi-Agent Clinical Physio Diagnostic System using smolagents, oMLX, and Langfuse.
Specialized for muscle, tendon, and peripheral nerve pathologies.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from langfuse import Langfuse
from sentence_transformers import SentenceTransformer
from smolagents import CodeAgent, DuckDuckGoSearchTool, ManagedAgent, OpenAIServerModel, tool

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------
LANGFUSE_SECRET_KEY = "sk-lf-2a24bee0-e7ca-4f88-8d9e-3d511708d502"
LANGFUSE_PUBLIC_KEY = "pk-lf-ccc486e0-0d3c-4245-9f5f-987353d79cea"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"

OMLX_API_KEY = "sk-omlx-gWACPPkjsP67zAEgJ3GReJ6q"
OMLX_BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-mlx-4Bit"

os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_BASE_URL,
)

# ---------------------------------------------------------------------------
# Shared LLM Backbone
# ---------------------------------------------------------------------------
omlx_model = OpenAIServerModel(
    model_id=MODEL_ID,
    api_base=OMLX_BASE_URL,
    api_key=OMLX_API_KEY,
)

# ---------------------------------------------------------------------------
# Local Vector Memory Module
# ---------------------------------------------------------------------------
class LocalVectorMemory:
    """Lightweight in-memory vector store using local sentence embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.corpus: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        new_vec = self.encoder.encode([text], normalize_embeddings=True)
        self.corpus.append(text)
        self.metadata.append(meta or {})
        if self.embeddings is None:
            self.embeddings = new_vec
        else:
            self.embeddings = np.vstack([self.embeddings, new_vec])

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.corpus) == 0:
            return []
        query_vec = self.encoder.encode([text], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "text": self.corpus[idx],
                "meta": self.metadata[idx]
            })
        return results

memory_store = LocalVectorMemory()

# ---------------------------------------------------------------------------
# Custom smolagents Tools
# ---------------------------------------------------------------------------
@tool
def store_case_memory(case_summary: str, diagnosis_or_findings: str) -> str:
    """Store intermediate clinical findings, differential hypotheses, and patient test results.
    
    Args:
        case_summary: Condensed summary of symptoms, physical test maneuvers performed, and findings.
        diagnosis_or_findings: Current differential hypothesis and suspected anatomical structures.
    """
    record = f"Findings: {case_summary} | Differential: {diagnosis_or_findings}"
    memory_store.add(record, meta={"summary": case_summary, "findings": diagnosis_or_findings})
    return "Successfully memorized intermediate case context."


@tool
def retrieve_similar_cases(query: str) -> str:
    """Retrieve past case interactions, anatomical patterns, or physical test findings.
    
    Args:
        query: Symptom patterns, nerve/tendon/muscle keywords, or clinical tests to recall.
    """
    matches = memory_store.query(query, top_k=2)
    if not matches:
        return "No relevant historical cases or consolidated notes found in memory."
    return "\n".join([f"- [Sim: {m['score']:.2f}] {m['text']}" for m in matches])


@tool
def evaluate_diagnostic_certainty(
    leading_diagnosis: str,
    confidence_score: float,
    rationale: str,
    recommended_provocative_test: str
) -> str:
    """Calibrate clinical certainty. If confidence is below 0.85, output the next provocative test.
    
    Args:
        leading_diagnosis: The primary suspected pathology (e.g., Supraspinatus Tendinopathy, Radial Tunnel Syndrome).
        confidence_score: Confidence level between 0.0 and 1.0 based on clinical evidence.
        rationale: Clinical reasoning connecting the symptoms and physical tests.
        recommended_provocative_test: Next physical test or targeted question if confidence < 0.85.
    """
    return json.dumps({
        "leading_diagnosis": leading_diagnosis,
        "confidence": confidence_score,
        "is_confident": confidence_score >= 0.85,
        "rationale": rationale,
        "next_step": "Finalize diagnosis" if confidence_score >= 0.85 else recommended_provocative_test
    }, indent=2)

# ---------------------------------------------------------------------------
# Multi-Agent Architecture
# ---------------------------------------------------------------------------
def build_multi_agent_system() -> CodeAgent:
    # 1. Specialized Literature & Evidence Retrieval Sub-Agent
    research_agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=omlx_model,
        max_steps=4,
        planning_interval=None,
    )

    managed_researcher = ManagedAgent(
        agent=research_agent,
        name="literature_researcher",
        description=(
            "Expert physio research agent. Delegates web searches to verify rare muscle, "
            "tendon, and nerve entrapment syndromes, differential diagnostic criteria, "
            "and sensitivity/specificity of provocative orthopedic tests. Returns concise summaries."
        ),
    )

    # 2. Primary Clinical Diagnostic Agent
    clinical_agent = CodeAgent(
        tools=[store_case_memory, retrieve_similar_cases, evaluate_diagnostic_certainty],
        managed_agents=[managed_researcher],
        model=omlx_model,
        max_steps=6,
    )
    return clinical_agent

# ---------------------------------------------------------------------------
# Interactive Patient Grilling Mode (VS Code Terminal)
# ---------------------------------------------------------------------------
def run_interactive_patient_session() -> None:
    print("\n" + "=" * 70)
    print("Physiotherapy Diagnostic Multi-Agent System (Interactive CLI)")
    print("Type 'exit' to quit at any time.")
    print("=" * 70 + "\n")
    
    agent = build_multi_agent_system()
    session_history: List[str] = []
    trace = langfuse_client.trace(name="interactive_physio_consultation")

    print("Agent: Hello. I specialize in complex musculoskeletal and neurological conditions.")
    patient_input = input("Describe your initial symptoms and onset: ")
    session_history.append(f"Patient Initial: {patient_input}")

    while True:
        if patient_input.strip().lower() in ["exit", "quit"]:
            break

        task_prompt = (
            "You are a rigorous clinical physiotherapy specialist diagnosing muscle, tendon, "
            "and nerve conditions. Review the consultation history below:\n\n"
            f"{chr(10).join(session_history)}\n\n"
            "Instructions:\n"
            "1. Delegate to 'literature_researcher' if differentiating between confounding pathologies.\n"
            "2. Retrieve memory to cross-reference previous findings.\n"
            "3. Call 'evaluate_diagnostic_certainty' with your estimated certainty (0.0 to 1.0).\n"
            "4. If confidence is < 0.85, do NOT diagnose yet. 'Grill' the patient: ask them to perform "
            "a specific provocative physical test (e.g., Spurling, Phalen, resisted isometric contraction, "
            "palpation) and report their response.\n"
            "5. If confidence is >= 0.85, provide the final clinical diagnosis, anatomical structures, "
            "and recovery protocol."
        )

        span = trace.span(name="consultation_step", input={"prompt": task_prompt})
        response = agent.run(task_prompt)
        span.end(output={"response": str(response)})

        print(f"\n[Physio Agent]:\n{response}\n")

        # Check for confirmed diagnosis
        if "confidence: 0.8" in str(response) or "is_confident\": true" in str(response).lower():
            print("--- Diagnostic certainty achieved. Session concluded. ---")
            break

        patient_input = input("Your response / test outcome: ")
        session_history.append(f"Patient Reply: {patient_input}")

    trace.update(output={"status": "completed"})
    langfuse_client.flush()

# ---------------------------------------------------------------------------
# Offline 1,000 Medical MCQA Benchmark Mode
# ---------------------------------------------------------------------------
def run_offline_benchmark(sample_limit: int = 1000) -> None:
    print(f"\nLoading HF MedMCQA dataset (evaluating up to {sample_limit} questions)...")
    dataset = load_dataset("medmcqa", split="validation", streaming=True)
    
    agent = build_multi_agent_system()
    trace = langfuse_client.trace(name="offline_medmcqa_benchmark")

    keywords = ["muscle", "tendon", "nerve", "paresthesia", "plexus", "radiculopathy", "joint", "motor"]
    evaluated_count = 0
    correct_count = 0

    for item in dataset:
        if evaluated_count >= sample_limit:
            break

        question = item.get("question", "")
        # Filter for relevant neuro-musculoskeletal questions
        if not any(k in question.lower() for k in keywords):
            continue

        options = {
            "A": item.get("opa", ""),
            "B": item.get("opb", ""),
            "C": item.get("opc", ""),
            "D": item.get("opd", "")
        }
        
        # MedMCQA uses 1-based index (1->A, 2->B, etc.) or 0-based depending on split
        cop = item.get("cop")
        cop_letter = ["A", "B", "C", "D"][cop] if isinstance(cop, int) and 0 <= cop <= 3 else "A"

        mcq_prompt = (
            f"Question: {question}\n"
            f"A: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}\n\n"
            "Analyze the anatomical structures involved (muscles, tendons, nerves). "
            "Use the 'literature_researcher' if needed to verify conflicting options. "
            "State your step-by-step reasoning, and end your answer with 'FINAL ANSWER: <Option Letter>'."
        )

        span = trace.span(name="mcq_evaluation", input={"question": question, "options": options})
        result = str(agent.run(mcq_prompt))
        
        match = re.search(r"FINAL ANSWER:\s*([A-D])", result, re.IGNORECASE)
        predicted_letter = match.group(1).upper() if match else "UNKNOWN"
        is_correct = (predicted_letter == cop_letter)
        
        if is_correct:
            correct_count += 1
        evaluated_count += 1

        span.end(output={
            "prediction": predicted_letter,
            "correct_option": cop_letter,
            "is_correct": is_correct,
            "full_output": result
        })

        print(f"[{evaluated_count}/{sample_limit}] Correct: {is_correct} (Pred: {predicted_letter}, Truth: {cop_letter})")

    accuracy = (correct_count / evaluated_count) if evaluated_count > 0 else 0.0
    trace.update(output={"accuracy": accuracy, "total_evaluated": evaluated_count, "correct": correct_count})
    langfuse_client.flush()
    print(f"\nBenchmark Complete. Total Accuracy: {accuracy * 100:.2f}% ({correct_count}/{evaluated_count})")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    if mode == "benchmark":
        run_offline_benchmark(sample_limit=1000)
    else:
        run_interactive_patient_session()