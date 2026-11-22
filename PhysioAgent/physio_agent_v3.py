"""
Optimized Multi-Agent Clinical Physio Diagnostic & Benchmarking System.
Fast local inference on Apple Silicon (oMLX) + Langfuse Observability.
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
from langfuse import Langfuse
from sentence_transformers import SentenceTransformer
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

# ---------------------------------------------------------------------------
# Resilient DuckDuckGo Search Setup
# ---------------------------------------------------------------------------
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ---------------------------------------------------------------------------
# Observability & Model Credentials
# ---------------------------------------------------------------------------
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-2a24bee0-e7ca-4f88-8d9e-3d511708d502")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-ccc486e0-0d3c-4245-9f5f-987353d79cea")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

OMLX_API_KEY = os.getenv("OMLX_API_KEY", "sk-omlx-gWACPPkjsP67zAEgJ3GReJ6q")
OMLX_BASE_URL = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
MODEL_ID = "Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-mlx-4Bit"

os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

# Direct Langfuse Client
try:
    lf_client = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL,
    )
except Exception as e:
    print(f"[Warning] Langfuse initialization: {e}")
    lf_client = None

# ---------------------------------------------------------------------------
# LLM Model Client
# ---------------------------------------------------------------------------
def get_model(temperature: float = 0.0, max_tokens: int = 512) -> OpenAIServerModel:
    return OpenAIServerModel(
        model_id=MODEL_ID,
        api_base=OMLX_BASE_URL,
        api_key=OMLX_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# ---------------------------------------------------------------------------
# Lightweight Local Vector Memory
# ---------------------------------------------------------------------------
class CompactVectorMemory:
    """Fast, local vector store for symptom patterns and past diagnoses."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.records: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None

    def store(self, summary: str, diagnosis: str) -> None:
        text = f"{summary} -> {diagnosis}"
        emb = self.encoder.encode([text], normalize_embeddings=True)
        self.records.append({"text": text, "diagnosis": diagnosis})
        self.matrix = emb if self.matrix is None else np.vstack([self.matrix, emb])

    def query(self, text: str, top_k: int = 2) -> str:
        if self.matrix is None or len(self.records) == 0:
            return "No historical precedent found."
        q_emb = self.encoder.encode([text], normalize_embeddings=True)[0]
        scores = np.dot(self.matrix, q_emb)
        top_ids = np.argsort(scores)[::-1][:top_k]
        return " | ".join([f"[{scores[i]:.2f}] {self.records[i]['text']}" for i in top_ids if scores[i] > 0.4])

memory = CompactVectorMemory()

# ---------------------------------------------------------------------------
# Compact Tools (Strict Token Capping)
# ---------------------------------------------------------------------------
@tool
def search_medical_literature(query: str) -> str:
    """Fast web search for physical therapy tests, sensitivity/specificity, or nerve/muscle differentials.

    Args:
        query: Specific test name or condition (e.g., 'FAIR test sensitivity piriformis', 'Spurling test accuracy').
    """
    if DDGS is None:
        return "Search unavailable."
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=2):
                snippet = (r.get("body") or "")[:250].replace("\n", " ")
                results.append(f"{r.get('title')}: {snippet}")
        return "\n".join(results) if results else "No specific evidence found."
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def check_precedent_cases(symptoms: str) -> str:
    """Checks vector memory for similar muscle/tendon/nerve symptoms and past confirmed diagnoses.

    Args:
        symptoms: Clinical summary of pain locations and provocative test outcomes.
    """
    return memory.query(symptoms, top_k=2)

# ---------------------------------------------------------------------------
# Consultation State Container
# ---------------------------------------------------------------------------
@dataclass
class ClinicalState:
    complaint: str = ""
    tests: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def summarize(self) -> str:
        tests_str = "; ".join(self.tests) if self.tests else "None performed yet"
        return f"Complaint: {self.complaint} | Tests: {tests_str} | Current Confidence: {self.confidence:.2f}"

# ---------------------------------------------------------------------------
# Fast Multi-Agent Factory
# ---------------------------------------------------------------------------
def build_clinician_agent() -> ToolCallingAgent:
    """Lightweight ToolCallingAgent (avoids code execution timeouts of CodeAgent)."""
    return ToolCallingAgent(
        tools=[search_medical_literature, check_precedent_cases],
        model=get_model(temperature=0.1, max_tokens=600),
        max_steps=2,
    )

# ---------------------------------------------------------------------------
# Output Extraction Helper
# ---------------------------------------------------------------------------
def parse_response(text: str) -> Dict[str, Any]:
    """Extracts structured JSON payload or applies regex fallbacks."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{[\s\S]*\"confidence\"[\s\S]*\})", text)

    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass

    conf_match = re.search(r"confidence[\"':\s]+([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    conf = float(conf_match.group(1)) if conf_match else 0.40
    if conf > 1.0:
        conf = conf / 100.0

    return {
        "confidence": conf,
        "status": "CONFIRMED" if conf >= 0.85 else "GRILL",
        "question_or_test": text.strip()[:400],
        "diagnosis": "Clinical Diagnosis" if conf >= 0.85 else None,
        "rationale": "Direct evaluation",
    }

# ---------------------------------------------------------------------------
# Interactive Consultation Mode
# ---------------------------------------------------------------------------
def run_interactive_consultation() -> None:
    print("\n" + "=" * 70)
    print("PHYSIO MULTI-AGENT DIAGNOSTIC SYSTEM (FAST MODE)")
    print("=" * 70)
    print("Specializing in muscles, tendons, and nerve entrapments.")
    print("Type 'exit' to quit.\n")

    initial_input = input("Lead Clinician: What symptoms bring you in today? Describe onset and pain:\n> ").strip()
    if not initial_input or initial_input.lower() in ["exit", "quit"]:
        return

    agent = build_clinician_agent()
    state = ClinicalState(complaint=initial_input)

    # Langfuse Trace
    trace = None
    if lf_client:
        try:
            trace = lf_client.trace(
                name="physio_interactive_consultation",
                metadata={"model": MODEL_ID, "complaint": initial_input[:50]},
            )
        except Exception as e:
            print(f"[Langfuse Trace Error]: {e}")

    turn = 1
    MAX_TURNS = 6
    CONFIDENCE_THRESHOLD = 0.85

    while turn <= MAX_TURNS:
        print(f"\n[Turn {turn}] Clinician is reasoning...")
        
        prompt = (
            "You are a clinical physical therapy specialist in muscle, tendon, and peripheral nerve disorders.\n"
            f"SUMMARY: {state.summarize()}\n\n"
            "INSTRUCTIONS:\n"
            "1. Use 'search_medical_literature' ONLY if you need to confirm test specificity or differential.\n"
            "2. Decide your confidence (0.0 to 1.0).\n"
            "3. Return ONLY a JSON object:\n"
            "```json\n"
            "{\n"
            '  "confidence": 0.45,\n'
            '  "status": "GRILL" or "CONFIRMED",\n'
            '  "question_or_test": "Instructions for next physical test or question",\n'
            '  "diagnosis": null or "Confirmed condition name",\n'
            '  "rationale": "Brief 1-sentence reasoning"\n'
            "}\n"
            "```\n"
            f"If confidence < {CONFIDENCE_THRESHOLD}, status MUST be 'GRILL'. Demand the patient perform a specific physical test."
        )

        span = None
        if trace:
            try:
                span = trace.span(name=f"turn_{turn}", input={"state": asdict(state)})
            except Exception:
                pass

        try:
            raw_output = str(agent.run(prompt))
        except Exception as err:
            raw_output = f"Inference note: {str(err)}"

        parsed = parse_response(raw_output)
        conf = float(parsed.get("confidence", 0.4))
        state.confidence = conf
        status = str(parsed.get("status", "GRILL")).upper()
        patient_step = parsed.get("question_or_test", raw_output)
        rationale = parsed.get("rationale", "")

        if span:
            try:
                span.end(output={"confidence": conf, "parsed": parsed})
            except Exception:
                pass

        if status == "CONFIRMED" or conf >= CONFIDENCE_THRESHOLD:
            final_dx = parsed.get("diagnosis") or "Definitive Physiotherapy Diagnosis"
            print(f"\n Lead Clinician [Confidence: {conf*100:.1f}%]:")
            print(f"**Final Diagnosis**: {final_dx}")
            print(f"**Rationale**: {rationale}\n")
            memory.store(state.summarize(), final_dx)
            if trace:
                trace.update(output={"diagnosis": final_dx, "confidence": conf})
            break

        print(f"\n Lead Clinician [Confidence: {conf*100:.1f}%]:\n{patient_step}\n")

        user_reply = input("Your answer / test result:\n> ").strip()
        if not user_reply or user_reply.lower() in ["exit", "quit"]:
            print("Session ended by patient.")
            break

        state.tests.append(f"{patient_step[:40]} -> {user_reply}")
        turn += 1

    if lf_client:
        try:
            lf_client.flush()
            print("[Langfuse] Trace successfully flushed.")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Fast Offline Benchmark (100 Medical QA Items)
# ---------------------------------------------------------------------------
def run_offline_benchmark(sample_limit: int = 100) -> None:
    print("\n" + "=" * 70)
    print(f"OFFLINE PHYSIO BENCHMARK: MedQA USMLE (Target: {sample_limit} items)")
    print("=" * 70)

    # Use direct model generation for benchmark speed (avoids tool loops)
    model = get_model(temperature=0.0, max_tokens=150)

    print("Streaming dataset from 'GBaker/MedQA-USMLE-4-options'...")
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", streaming=True)
    except Exception as e:
        print(f"Error connecting to dataset: {e}. Using local benchmark cases.")
        dataset = [
            {
                "question": "A 35-year-old athlete has deep gluteal pain reproduced with hip flexion, adduction, and internal rotation (FAIR test). Which structure is compressing the sciatic nerve?",
                "options": {"A": "Obturator externus", "B": "Piriformis muscle", "C": "Quadratus femoris", "D": "Gluteus medius"},
                "answer_idx": "B",
            },
            {
                "question": "A typist has tingling in the thumb, index, and middle finger. Phalen and Tinel signs are positive over the volar wrist. What nerve is affected?",
                "options": {"A": "Ulnar nerve", "B": "Radial nerve", "C": "Median nerve", "D": "Axillary nerve"},
                "answer_idx": "C",
            },
            {
                "question": "Pain with resisted thumb extension and positive Finkelstein test over the radial styloid process indicates tenosynovitis of which tendons?",
                "options": {"A": "Abductor pollicis longus and extensor pollicis brevis", "B": "Flexor carpi radialis", "C": "Extensor indicis", "D": "Extensor carpi ulnaris"},
                "answer_idx": "A",
            },
        ]

    keywords = ["muscle", "tendon", "nerve", "pain", "radiculopathy", "plexus", "spine", "joint", "motor", "sensory", "reflex"]

    trace = None
    if lf_client:
        try:
            trace = lf_client.trace(
                name="offline_medqa_benchmark",
                metadata={"model": MODEL_ID, "limit": sample_limit},
            )
        except Exception:
            pass

    evaluated = 0
    correct = 0

    for item in dataset:
        if evaluated >= sample_limit:
            break

        question = item.get("question") or ""
        if not any(k in question.lower() for k in keywords):
            continue

        # Extract options
        opts_raw = item.get("options")
        if isinstance(opts_raw, dict):
            opts = opts_raw
        elif isinstance(opts_raw, list):
            opts = {chr(65 + i): opt for i, opt in enumerate(opts_raw[:4])}
        else:
            opts = {"A": item.get("opa", "A"), "B": item.get("opb", "B"), "C": item.get("opc", "C"), "D": item.get("opd", "D")}

        ground_truth = str(item.get("answer_idx") or item.get("cop") or "A").strip()
        if ground_truth in ["1", "2", "3", "4"]:
            ground_truth = {"1": "A", "2": "B", "3": "C", "4": "D"}[ground_truth]

        prompt = (
            f"Question: {question}\n"
            f"A: {opts.get('A', '')}\n"
            f"B: {opts.get('B', '')}\n"
            f"C: {opts.get('C', '')}\n"
            f"D: {opts.get('D', '')}\n\n"
            "Identify the correct option (A, B, C, or D). Answer in this format only: 'FINAL ANSWER: <Letter>'"
        )

        try:
            response = str(model(prompt))
            match = re.search(r"FINAL ANSWER:\s*([A-D])", response, re.IGNORECASE)
            pred = match.group(1).upper() if match else ("A" if "A" in response[:20] else "UNKNOWN")
        except Exception as e:
            pred = "ERROR"

        is_correct = (pred == ground_truth)
        if is_correct:
            correct += 1
        evaluated += 1

        if trace and lf_client:
            try:
                lf_client.score(
                    trace_id=trace.id,
                    name="accuracy",
                    value=1.0 if is_correct else 0.0,
                    comment=f"Q{evaluated}: Pred {pred} vs Truth {ground_truth}",
                )
            except Exception:
                pass

        print(f"[{evaluated:03d}/{sample_limit}] {'PASS' if is_correct else 'FAIL'} | Pred: {pred} | Truth: {ground_truth} | Acc: {(correct/evaluated)*100:.1f}%")

    final_acc = (correct / evaluated) if evaluated > 0 else 0.0
    if trace:
        try:
            trace.update(output={"accuracy": final_acc, "total": evaluated, "correct": correct})
        except Exception:
            pass

    if lf_client:
        try:
            lf_client.flush()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE: {correct}/{evaluated} ({final_acc*100:.1f}%)")
    print("=" * 70)

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "interactive"
    if mode in ["benchmark", "eval", "test"]:
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_offline_benchmark(sample_limit=limit)
    else:
        run_interactive_consultation()