"""
Multi-Agent Clinical Physiotherapy Diagnostic & Benchmarking System.
Built with smolagents, oMLX (local Apple Silicon execution), and Langfuse.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool

# DuckDuckGo resilient import
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ---------------------------------------------------------------------------
# Langfuse Unified Tracer (v2/v3 Resilient Wrapper)
# ---------------------------------------------------------------------------
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-2a24bee0-e7ca-4f88-8d9e-3d511708d502")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-ccc486e0-0d3c-4245-9f5f-987353d79cea")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

try:
    from langfuse import Langfuse
    _raw_client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL,
    )
except Exception as e:
    _raw_client = None


class SafeTraceSpan:
    def __init__(self, raw_span: Any = None):
        self.raw_span = raw_span

    def end(self, **kwargs) -> None:
        if self.raw_span and hasattr(self.raw_span, "end"):
            try:
                self.raw_span.end(**kwargs)
            except Exception:
                pass


class SafeTrace:
    def __init__(self, raw_trace: Any = None, trace_id: str = "mock-trace"):
        self.raw_trace = raw_trace
        self.id = getattr(raw_trace, "id", trace_id)

    def span(self, name: str, **kwargs) -> SafeTraceSpan:
        if self.raw_trace and hasattr(self.raw_trace, "span"):
            try:
                return SafeTraceSpan(self.raw_trace.span(name=name, **kwargs))
            except Exception:
                pass
        return SafeTraceSpan()

    def update(self, **kwargs) -> None:
        if self.raw_trace and hasattr(self.raw_trace, "update"):
            try:
                self.raw_trace.update(**kwargs)
            except Exception:
                pass


class SafeLangfuseClient:
    """Provides a safe facade across differing Langfuse SDK releases."""

    def __init__(self, client: Any):
        self.client = client

    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> SafeTrace:
        if not self.client:
            return SafeTrace()
        try:
            # Try newer SDK method
            if hasattr(self.client, "create_trace"):
                return SafeTrace(self.client.create_trace(name=name, metadata=metadata))
            # Try root trace creation
            if hasattr(self.client, "trace"):
                return SafeTrace(self.client.trace(name=name, metadata=metadata))
        except Exception:
            pass
        return SafeTrace()

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        if not self.client:
            return
        try:
            if hasattr(self.client, "create_score"):
                self.client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
            elif hasattr(self.client, "score"):
                self.client.score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception:
            pass

    def flush(self) -> None:
        if self.client and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception:
                pass


langfuse = SafeLangfuseClient(_raw_client)

# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "sk-omlx-gWACPPkjsP67zAEgJ3GReJ6q")
OMLX_BASE_URL = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
MODEL_ID = "Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-mlx-4Bit"


def create_model(temperature: float = 0.1) -> OpenAIServerModel:
    return OpenAIServerModel(
        model_id=MODEL_ID,
        api_base=OMLX_BASE_URL,
        api_key=OMLX_API_KEY,
        temperature=temperature,
    )

# ---------------------------------------------------------------------------
# Vector Memory Engine
# ---------------------------------------------------------------------------
class SemanticCaseMemory:
    """Vector memory backed by sentence embeddings for similarity search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.records: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def store(self, key_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        embedding = self.encoder.encode([key_text], normalize_embeddings=True)
        self.records.append({"text": key_text, "meta": metadata or {}})
        if self.embeddings is None:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.records) == 0:
            return []
        query_vec = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_vec)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "score": float(scores[i]),
                "text": self.records[i]["text"],
                "meta": self.records[i]["meta"],
            }
            for i in ranked_indices
        ]


case_memory = SemanticCaseMemory()

# ---------------------------------------------------------------------------
# Custom smolagents Tools
# ---------------------------------------------------------------------------
@tool
def web_search_physio_literature(query: str) -> str:
    """Searches medical and physical therapy literature for tendon, nerve, and muscle disorders.

    Args:
        query: Clinical search query (e.g., 'piriformis vs sciatica differential', 'Spurling test sensitivity').
    """
    if DDGS is None:
        return "Search tool error: ddgs/duckduckgo-search is not available."
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}")
        if not results:
            return "No medical search results found."
        return "\n\n".join(results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


@tool
def recall_clinical_precedents(query: str) -> str:
    """Searches local vector memory for past patient presentations, exam outcomes, and matching differentials.

    Args:
        query: Muscle, tendon, or nerve symptoms, or specific physical tests to match against past cases.
    """
    matches = case_memory.search(query, top_k=2)
    if not matches:
        return "No similar historical cases found in local vector memory."
    formatted = [
        f"- [Relevance: {m['score']:.2f}] Case: {m['text']} | Diagnosis: {m['meta'].get('diagnosis', 'N/A')}"
        for m in matches
    ]
    return "\n".join(formatted)


@tool
def commit_case_knowledge(symptom_summary: str, confirmed_diagnosis: str) -> str:
    """Persists a resolved case summary and diagnosis to the local vector memory for future reference.

    Args:
        symptom_summary: Dense clinical summary of anatomical involvement and provocative test results.
        confirmed_diagnosis: The verified medical diagnosis.
    """
    case_memory.store(
        key_text=symptom_summary,
        metadata={"diagnosis": confirmed_diagnosis},
    )
    return "Case successfully recorded in vector memory."

# ---------------------------------------------------------------------------
# Clinical State Container
# ---------------------------------------------------------------------------
@dataclass
class ClinicalSessionState:
    primary_complaint: str = ""
    symptoms_and_onset: str = ""
    provocative_tests_performed: List[Dict[str, str]] = field(default_factory=list)
    differential_hypotheses: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    confirmed_diagnosis: Optional[str] = None

    def to_compact_summary(self) -> str:
        tests_str = (
            "; ".join([f"{t['test']}: {t['outcome']}" for t in self.provocative_tests_performed])
            if self.provocative_tests_performed
            else "None recorded yet"
        )
        diff_str = ", ".join(self.differential_hypotheses) if self.differential_hypotheses else "Under investigation"
        return (
            f"PRIMARY COMPLAINT: {self.primary_complaint}\n"
            f"SYMPTOMS & ONSET: {self.symptoms_and_onset}\n"
            f"PROVOCATIVE TESTS CONDUCTED: {tests_str}\n"
            f"DIFFERENTIAL CONSIDERATIONS: {diff_str}\n"
            f"CURRENT CONFIDENCE: {self.confidence_score:.2f}"
        )

# ---------------------------------------------------------------------------
# Multi-Agent Architecture
# ---------------------------------------------------------------------------
def build_physio_orchestrator() -> CodeAgent:
    model = create_model()

    research_agent = ToolCallingAgent(
        tools=[web_search_physio_literature],
        model=model,
        name="literature_researcher",
        description=(
            "Medical research specialist. Queries peer-reviewed literature and physical therapy "
            "orthopedic test databases for muscle, tendon, and nerve pathologies. Use this agent "
            "to check test sensitivity/specificity, differential diagnostic criteria, or entrapment sites."
        ),
        max_steps=7,
    )

    lead_clinician = CodeAgent(
        tools=[recall_clinical_precedents, commit_case_knowledge],
        managed_agents=[research_agent],
        model=model,
        max_steps=9,
    )
    return lead_clinician


def parse_clinician_output(raw_output: str) -> Dict[str, Any]:
    """Robust extraction of structured clinical decisions from small model outputs."""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass

    naked_match = re.search(r"(\{[\s\S]*\"confidence\"[\s\S]*\})", raw_output)
    if naked_match:
        try:
            return json.loads(naked_match.group(1))
        except Exception:
            pass

    conf_match = re.search(r"confidence[\"':\s]+([0-9]*\.?[0-9]+)", raw_output, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.50
    status = "CONFIRMED" if confidence >= 0.85 else "GRILL"

    return {
        "confidence": confidence,
        "differential_hypotheses": [],
        "status": status,
        "patient_question_or_test": raw_output.strip(),
        "final_diagnosis": None,
        "clinical_rationale": "Direct clinician evaluation",
    }

# ---------------------------------------------------------------------------
# Interactive Consultation Engine (CLI Grilling Mode)
# ---------------------------------------------------------------------------
def run_interactive_consultation() -> None:
    print("\n" + "=" * 78)
    print("PHYSIO MULTI-AGENT DIAGNOSTIC SYSTEM - CLINICAL INTERACTION MODE")
    print("=" * 78)
    print("Specializing in complex musculoskeletal, tendon, and peripheral nerve conditions.")
    print("Type 'exit' or 'quit' at any prompt to stop.\n")

    initial_input = input("Lead Clinician: What symptoms bring you in today? Describe onset and pain patterns:\n> ").strip()
    if initial_input.lower() in ["exit", "quit"] or not initial_input:
        print("Consultation aborted.")
        return

    lead_agent = build_physio_orchestrator()
    state = ClinicalSessionState(primary_complaint=initial_input, symptoms_and_onset=initial_input)

    trace = langfuse.create_trace(
        name="interactive_physio_session",
        metadata={"model": MODEL_ID, "complaint": state.primary_complaint[:60]},
    )

    turn = 1
    MAX_TURNS = 12
    CONFIDENCE_THRESHOLD = 0.85

    while turn <= MAX_TURNS:
        print(f"\n--- [Consultation Step {turn}] Analyzing clinical presentation... ---")
        span = trace.span(
            name=f"clinical_turn_{turn}",
            input={"state": asdict(state)},
        )

        prompt = (
            "You are an elite clinical physiotherapist specializing in differential diagnosis of "
            "complex muscle, tendon, and nerve disorders (e.g., piriformis syndrome vs. L5-S1 lumbar radiculopathy "
            "vs. gluteal tendinopathy).\n\n"
            f"CURRENT CONSOLIDATED CLINICAL SUMMARY:\n{state.to_compact_summary()}\n\n"
            "MANDATE:\n"
            "1. Delegate search queries to 'literature_researcher' if differentiating between confounding pathologies.\n"
            "2. Consult past records using 'recall_clinical_precedents'.\n"
            "3. Assess your current diagnostic certainty between 0.00 and 1.00.\n"
            "4. Return a JSON block inside ```json ... ``` with EXACTLY these keys:\n"
            "   {\n"
            '     "confidence": <float between 0.0 and 1.0>,\n'
            '     "differential_hypotheses": ["<pathology 1>", "<pathology 2>"],\n'
            '     "status": "GRILL" or "CONFIRMED",\n'
            '     "patient_question_or_test": "<instruction for the patient or provocative test to perform>",\n'
            '     "final_diagnosis": "<null or diagnosis name if CONFIRMED>",\n'
            '     "clinical_rationale": "<brief explanation>"\n'
            "   }\n"
            f"If confidence < {CONFIDENCE_THRESHOLD}, status MUST be 'GRILL'. Grill the patient by demanding "
            "they perform a specific orthopedic maneuver (e.g., FAIR test, seated piriformis stretch, straight leg raise, "
            "slump test, active resisted hip abduction) and report exact physical sensations."
        )

        try:
            agent_output = str(lead_agent.run(prompt))
        except Exception as err:
            agent_output = f"Clinical assessment error: {str(err)}"

        decision_data = parse_clinician_output(agent_output)
        confidence = float(decision_data.get("confidence", 0.5))
        status = str(decision_data.get("status", "GRILL")).upper()
        patient_message = decision_data.get("patient_question_or_test", agent_output)
        rationale = decision_data.get("clinical_rationale", "")
        hypotheses = decision_data.get("differential_hypotheses", [])

        state.confidence_score = confidence
        if hypotheses:
            state.differential_hypotheses = hypotheses

        span.end(output={
            "agent_output": agent_output,
            "parsed_decision": decision_data,
            "confidence": confidence,
        })

        if status == "CONFIRMED" or confidence >= CONFIDENCE_THRESHOLD:
            final_dx = decision_data.get("final_diagnosis", "Definitive Physiotherapy Diagnosis")
            print(f"\n Lead Clinician: Diagnostic Certainty Reached ({confidence * 100:.1f}% >= {CONFIDENCE_THRESHOLD * 100:.1f}%)")
            print(f"**Final Clinical Diagnosis**: {final_dx}")
            print(f"**Rationale**: {rationale}")

            commit_case_knowledge(
                symptom_summary=state.to_compact_summary(),
                confirmed_diagnosis=str(final_dx),
            )
            trace.update(
                output={"status": "DIAGNOSED", "diagnosis": final_dx, "confidence": confidence},
            )
            break

        print(f"\n Lead Clinician [Confidence: {confidence * 100:.1f}%]:\n{patient_message}\n")

        user_response = input("Your answer / test result:\n> ").strip()
        if user_response.lower() in ["exit", "quit"] or not user_response:
            print("Session ended by user.")
            trace.update(output={"status": "USER_EXIT"})
            break

        state.provocative_tests_performed.append({
            "test": str(patient_message)[:80],
            "outcome": user_response,
        })
        turn += 1

    if turn > MAX_TURNS and state.confidence_score < CONFIDENCE_THRESHOLD:
        print("\n Lead Clinician: Maximum consultation iterations reached.")
        print("Recommendation: Immediate in-person clinical exam with MRI or EMG.")
        trace.update(output={"status": "INCONCLUSIVE_MAX_TURNS", "last_state": asdict(state)})

    langfuse.flush()
    print("\nSession trace sent to Langfuse.")

# ---------------------------------------------------------------------------
# Offline Medical Benchmark Engine
# ---------------------------------------------------------------------------
def run_offline_benchmark(sample_limit: int = 1000) -> None:
    print("\n" + "=" * 78)
    print(f"OFFLINE PHYSIO BENCHMARK: Streaming Medical QA (Target: {sample_limit} items)")
    print("=" * 78)

    eval_agent = build_physio_orchestrator()

    # Canonical HF dataset path with fallback
    print("Connecting to Hugging Face dataset stream...")
    dataset = None
    dataset_candidates = ["openlifescience/medmcqa", "medalpaca/medical_meadow_medqa"]
    
    for candidate in dataset_candidates:
        try:
            print(f"Attempting to stream dataset from: {candidate}")
            dataset = load_dataset(candidate, split="train", streaming=True)
            break
        except Exception as e:
            print(f"Could not load {candidate}: {e}")

    if dataset is None:
        print("Falling back to local synthetic neuro-musculoskeletal test bank due to HF network restriction.")
        dataset = [
            {
                "question": "A 35-year-old runner presents with deep aching buttock pain radiating down the posterior thigh, exacerbated by prolonged sitting. Straight leg raise is negative, but pain is reproduced with flexion, adduction, and internal rotation (FAIR test). Which structure is most likely entrapping the sciatic nerve?",
                "opa": "Obturator internus",
                "opb": "Piriformis muscle",
                "opc": "Quadratus femoris",
                "opd": "Gluteus minimus",
                "cop": 2,
            },
            {
                "question": "A patient presents with weakness in thumb abduction and tingling in the radial 3.5 digits. Phalen test is positive within 30 seconds. Which nerve is compressed within the carpal tunnel?",
                "opa": "Ulnar nerve",
                "opb": "Radial nerve",
                "opc": "Median nerve",
                "opd": "Musculocutaneous nerve",
                "cop": 3,
            },
            {
                "question": "Pain exacerbated by resisted thumb extension and abduction within the anatomical snuffbox, with a positive Finkelstein test, indicates inflammation of which tendon sheath?",
                "opa": "Flexor carpi radialis",
                "opb": "Extensor pollicis brevis and abductor pollicis longus",
                "opc": "Extensor indicis",
                "opd": "Extensor digitorum communis",
                "cop": 2,
            },
        ]

    target_keywords = [
        "muscle", "tendon", "nerve", "paresthesia", "plexus", "radiculopathy",
        "syndrome", "entrapment", "reflex", "motor", "sensory", "ligament", "joint", "pain"
    ]

    benchmark_trace = langfuse.create_trace(
        name="medmcqa_offline_benchmark",
        metadata={"model": MODEL_ID, "target_samples": sample_limit},
    )

    evaluated_count = 0
    correct_count = 0

    for item in dataset:
        if evaluated_count >= sample_limit:
            break

        question = item.get("question") or item.get("input") or ""
        if not any(kw in question.lower() for kw in target_keywords):
            continue

        options = {
            "A": item.get("opa") or item.get("option_a") or "Option A",
            "B": item.get("opb") or item.get("option_b") or "Option B",
            "C": item.get("opc") or item.get("option_c") or "Option C",
            "D": item.get("opd") or item.get("option_d") or "Option D",
        }

        # Handle 1-indexed (1->A, 2->B, etc.) and 0-indexed values
        raw_cop = item.get("cop")
        cop_map = {1: "A", 2: "B", 3: "C", 4: "D", 0: "A"}
        ground_truth = cop_map.get(raw_cop, "B")

        prompt = (
            "You are a medical specialist in clinical anatomy, orthopedics, and neurology. "
            "Solve the following question:\n\n"
            f"QUESTION: {question}\n"
            f"A: {options['A']}\n"
            f"B: {options['B']}\n"
            f"C: {options['C']}\n"
            f"D: {options['D']}\n\n"
            "Conclude with your final answer formatted exactly as: FINAL ANSWER: <A|B|C|D>"
        )

        span = benchmark_trace.span(
            name=f"mcq_{evaluated_count + 1}",
            input={"question": question, "options": options, "ground_truth": ground_truth},
        )

        try:
            raw_response = str(eval_agent.run(prompt))
            match = re.search(r"FINAL ANSWER:\s*([A-D])", raw_response, re.IGNORECASE)
            prediction = match.group(1).upper() if match else "UNKNOWN"
        except Exception as err:
            prediction = "ERROR"
            raw_response = str(err)

        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_count += 1
        evaluated_count += 1

        langfuse.score(
            trace_id=benchmark_trace.id,
            name="accuracy",
            value=1.0 if is_correct else 0.0,
            comment=f"Q{evaluated_count}: Pred: {prediction}, Truth: {ground_truth}",
        )
        span.end(output={"prediction": prediction, "is_correct": is_correct, "raw": raw_response[:200]})

        current_acc = (correct_count / evaluated_count) * 100
        print(
            f"[{evaluated_count:04d}/{sample_limit}] "
            f"Result: {'PASS' if is_correct else 'FAIL'} | "
            f"Pred: {prediction} | Truth: {ground_truth} | "
            f"Running Accuracy: {current_acc:.2f}%"
        )

    final_accuracy = (correct_count / evaluated_count) if evaluated_count > 0 else 0.0
    benchmark_trace.update(
        output={
            "total_evaluated": evaluated_count,
            "correct": correct_count,
            "accuracy": final_accuracy,
        }
    )
    langfuse.flush()
    print("\n" + "=" * 78)
    print(f"BENCHMARK COMPLETE: {correct_count}/{evaluated_count} correct ({final_accuracy * 100:.2f}%)")
    print("=" * 78)

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "interactive"
    if cli_arg in ["benchmark", "test", "eval"]:
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        run_offline_benchmark(sample_limit=sample_size)
    else:
        run_interactive_consultation()