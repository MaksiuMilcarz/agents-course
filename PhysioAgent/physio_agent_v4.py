"""
Multi-Agent Clinical Physiotherapy Diagnostic & Benchmarking System.
High-accuracy clinical differential engine with Langfuse tracing and local oMLX inference.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Resilient DDGS import
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ---------------------------------------------------------------------------
# Langfuse & Model Configuration
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

# Direct OpenAI client connected to oMLX
llm_client = OpenAI(
    base_url=OMLX_BASE_URL,
    api_key=OMLX_API_KEY,
)

# Robust Langfuse client initialization
try:
    from langfuse import Langfuse
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL,
    )
    # Validate connection
    langfuse.auth_check()
    print("[Observability] Langfuse cloud connected successfully.")
except Exception as e:
    print(f"[Observability Warning] Langfuse initialization warning: {e}")
    langfuse = None

# ---------------------------------------------------------------------------
# Vector Memory Engine (Anatomical & Past Case Store)
# ---------------------------------------------------------------------------
class ClinicalVectorStore:
    """Stores case summaries and retrieves anatomical test precedents."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.cases: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

    def add_case(self, case_summary: str, diagnosis: str) -> None:
        record = f"{case_summary} | Diagnosis: {diagnosis}"
        vec = self.encoder.encode([record], normalize_embeddings=True)
        self.cases.append({"text": record, "diagnosis": diagnosis})
        if self.vectors is None:
            self.vectors = vec
        else:
            self.vectors = np.vstack([self.vectors, vec])

    def query_precedents(self, query: str, top_k: int = 2) -> str:
        if self.vectors is None or len(self.cases) == 0:
            return "No prior recorded cases match this profile."
        q_vec = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.vectors, q_vec)
        best_indices = np.argsort(scores)[::-1][:top_k]
        matches = [f"• ({scores[i]:.2f}) {self.cases[i]['text']}" for i in best_indices if scores[i] > 0.45]
        return "\n".join(matches) if matches else "No closely matching precedent found."

memory = ClinicalVectorStore()

# ---------------------------------------------------------------------------
# Evidence Retrieval Engine (DuckDuckGo with Strict Summarization)
# ---------------------------------------------------------------------------
def query_medical_web(query: str) -> str:
    """Searches medical web resources and extracts concise clinical findings."""
    if DDGS is None:
        return "Search engine unavailable."
    try:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(f"physiotherapy clinical test {query}", max_results=2):
                body = (item.get("body") or "")[:260].replace("\n", " ")
                results.append(f"[{item.get('title')}]: {body}")
        return "\n".join(results) if results else "No specific medical literature returned."
    except Exception as err:
        return f"Literature retrieval notice: {str(err)}"

# ---------------------------------------------------------------------------
# Core LLM Generation with Fallback
# ---------------------------------------------------------------------------
def call_llm(messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 512) -> str:
    """Direct chat completion call to local oMLX endpoint."""
    response = llm_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""

# ---------------------------------------------------------------------------
# Consultation State Management
# ---------------------------------------------------------------------------
@dataclass
class ClinicalSession:
    chief_complaint: str = ""
    reported_symptoms: str = ""
    completed_tests: List[Dict[str, str]] = field(default_factory=list)
    tested_maneuvers: List[str] = field(default_factory=list)
    active_differentials: List[str] = field(default_factory=list)
    ruled_out_conditions: List[str] = field(default_factory=list)
    current_confidence: float = 0.0

    def get_summary(self) -> str:
        completed = (
            "\n".join([f"  - {t['test']}: {t['result']}" for t in self.completed_tests])
            if self.completed_tests
            else "  - None performed yet"
        )
        tested_names = ", ".join(self.tested_maneuvers) if self.tested_maneuvers else "None"
        diffs = ", ".join(self.active_differentials) if self.active_differentials else "Pending initial differentiation"
        ruled = ", ".join(self.ruled_out_conditions) if self.ruled_out_conditions else "None yet"

        return (
            f"CHIEF COMPLAINT: {self.chief_complaint}\n"
            f"SYMPTOMS: {self.reported_symptoms}\n"
            f"PREVIOUSLY CONDUCTED TESTS (DO NOT REPEAT):\n{completed}\n"
            f"ALREADY TESTED MANEUVERS: {tested_names}\n"
            f"ACTIVE DIFFERENTIALS UNDER INVESTIGATION: {diffs}\n"
            f"RULED OUT PATHOLOGIES: {ruled}\n"
            f"CURRENT CERTAINTY: {self.current_confidence:.2f}"
        )

# ---------------------------------------------------------------------------
# Interactive Grilling Agent (Anti-Circular & Evidence-Grounded)
# ---------------------------------------------------------------------------
def run_interactive_consultation() -> None:
    print("\n" + "=" * 76)
    print("PHYSIOTHERAPY CLINICAL REASONING SYSTEM - INTERACTIVE DIAGNOSTIC MODE")
    print("=" * 76)
    print("Specializing in peripheral nerve entrapments, tendinopathies, and myopathies.")
    print("Type 'exit' to quit.\n")

    complaint = input("Lead Clinician: What symptoms are you experiencing? Describe location and onset:\n> ").strip()
    if not complaint or complaint.lower() in ["exit", "quit"]:
        return

    session = ClinicalSession(chief_complaint=complaint, reported_symptoms=complaint)

    # Langfuse Trace
    trace = None
    if langfuse:
        try:
            trace = langfuse.trace(
                name="interactive_physio_consultation",
                metadata={"model": MODEL_ID, "complaint": complaint[:60]},
                input={"chief_complaint": complaint},
            )
        except Exception as e:
            print(f"[Langfuse notice]: {e}")

    turn = 1
    MAX_TURNS = 7
    CONFIDENCE_THRESHOLD = 0.85

    while turn <= MAX_TURNS:
        print(f"\n[Turn {turn}] Clinician is formulating differential analysis...")

        # 1. Check for relevant precedents in vector store
        precedents = memory.query_precedents(f"{session.chief_complaint} {session.reported_symptoms}")

        # 2. Targeted search if multiple confounding pathologies exist
        search_evidence = ""
        if turn >= 2 and len(session.active_differentials) > 1:
            search_query = f"{session.active_differentials[0]} vs {session.active_differentials[1]} physical examination"
            search_evidence = query_medical_web(search_query)

        # 3. Formulate Prompt with Strict Constraints to Eliminate Circular Reasoning
        system_prompt = (
            "You are an expert orthopedic and neurological physiotherapist. Your goal is to reach a "
            "definitive diagnosis (>0.85 confidence) or eliminate lookalike conditions.\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER repeat a test that has already been performed or tested.\n"
            "2. Each question or provocative test MUST gather new information to eliminate one specific differential.\n"
            "3. If previous test was positive, isolate the anatomical structure. If negative, pivot to the alternative.\n"
            "4. Output ONLY valid JSON inside ```json ... ``` with no conversational filler outside the block."
        )

        user_prompt = (
            f"CLINICAL STATE:\n{session.get_summary()}\n\n"
            f"VECTOR MEMORY PRECEDENTS:\n{precedents}\n\n"
            f"LITERATURE EVIDENCE:\n{search_evidence or 'Not needed for this step.'}\n\n"
            "Provide your clinical evaluation in the following JSON format:\n"
            "```json\n"
            "{\n"
            '  "confidence": <float from 0.00 to 1.00>,\n'
            '  "active_differentials": ["<condition 1>", "<condition 2>"],\n'
            '  "ruled_out": ["<condition ruled out>"],\n'
            '  "status": "GRILL" or "CONFIRMED",\n'
            '  "test_name": "<exact name of the orthopedic test to perform>",\n'
            '  "instructions_for_patient": "<clear step-by-step physical test for the patient to perform and describe>",\n'
            '  "final_diagnosis": null or "<Definitive Diagnosis Name>",\n'
            '  "clinical_rationale": "<1 sentence reasoning explaining what this test rules in or out>"\n'
            "}\n"
            "```"
        )

        span = None
        if trace:
            try:
                span = trace.span(
                    name=f"clinical_turn_{turn}",
                    input={"state": asdict(session), "prompt": user_prompt},
                )
            except Exception:
                pass

        raw_reply = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=650,
        )

        # Parse structured output
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_reply, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{[\s\S]*\"confidence\"[\s\S]*\})", raw_reply)

        decision: Dict[str, Any] = {}
        if json_match:
            try:
                decision = json.loads(json_match.group(1))
            except Exception:
                pass

        # Update clinical state from parsed output
        confidence = float(decision.get("confidence", 0.40))
        session.current_confidence = confidence
        if decision.get("active_differentials"):
            session.active_differentials = decision["active_differentials"]
        if decision.get("ruled_out"):
            session.ruled_out_conditions.extend([r for r in decision["ruled_out"] if r not in session.ruled_out_conditions])

        test_name = decision.get("test_name", "Clinical Examination")
        patient_instructions = decision.get("instructions_for_patient", raw_reply)
        rationale = decision.get("clinical_rationale", "")
        status = decision.get("status", "GRILL").upper()

        if span:
            try:
                span.end(output={"decision": decision, "confidence": confidence, "raw": raw_reply[:300]})
            except Exception:
                pass

        # Check for diagnosis confirmation
        if status == "CONFIRMED" or confidence >= CONFIDENCE_THRESHOLD:
            diagnosis = decision.get("final_diagnosis") or "Verified Neuro-Musculoskeletal Disorder"
            print(f"\n Lead Clinician [Diagnostic Confidence: {confidence * 100:.1f}%]:")
            print(f"**Final Clinical Diagnosis**: {diagnosis}")
            print(f"**Clinical Rationale**: {rationale}\n")

            memory.add_case(case_summary=session.get_summary(), diagnosis=diagnosis)
            if trace:
                trace.update(output={"diagnosis": diagnosis, "confidence": confidence, "status": "CONFIRMED"})
            break

        print(f"\n Lead Clinician [Confidence: {confidence * 100:.1f}% | Differentials: {', '.join(session.active_differentials)}]:")
        print(f"{patient_instructions}")
        if rationale:
            print(f"*(Goal: {rationale})*")
        print()

        patient_feedback = input("Your test outcome / physical response:\n> ").strip()
        if not patient_feedback or patient_feedback.lower() in ["exit", "quit"]:
            print("Consultation concluded by user.")
            if trace:
                trace.update(output={"status": "USER_ABORT"})
            break

        session.tested_maneuvers.append(test_name)
        session.completed_tests.append({"test": test_name, "result": patient_feedback})
        turn += 1

    if turn > MAX_TURNS and session.current_confidence < CONFIDENCE_THRESHOLD:
        print("\n Lead Clinician: Completed maximum testing threshold.")
        print(f"Working Differential: {', '.join(session.active_differentials)}")
        print("Recommendation: Refer for diagnostic imaging (MRI/Ultrasound) or Electromyography (EMG).")
        if trace:
            trace.update(output={"status": "INCONCLUSIVE", "confidence": session.current_confidence})

    if langfuse:
        try:
            langfuse.flush()
            print("[Observability] Consultation trace successfully sent to Langfuse Cloud.")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Offline Medical Benchmark Engine (High-Performance 100-Question Runner)
# ---------------------------------------------------------------------------
def run_offline_benchmark(sample_limit: int = 100) -> None:
    print("\n" + "=" * 76)
    print(f"OFFLINE PHYSIO BENCHMARK: MedQA USMLE Streaming (Target: {sample_limit} items)")
    print("=" * 76)

    print("Connecting to Hugging Face dataset stream ('GBaker/MedQA-USMLE-4-options')...")
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", streaming=True)
    except Exception as e:
        print(f"Unable to stream directly from Hub: {e}. Utilizing built-in neuro-muscular evaluation set.")
        dataset = [
            {
                "question": "A 40-year-old tennis player presents with severe lateral elbow pain worsened by resisted wrist extension. What structure is affected?",
                "options": {"A": "Extensor carpi radialis brevis tendon", "B": "Flexor carpi ulnaris", "C": "Biceps brachii tendon", "D": "Pronator teres"},
                "answer_idx": "A",
            },
            {
                "question": "A runner has deep buttock pain and tingling in the posterior thigh. Pain is provoked by passive hip flexion, adduction, and internal rotation (FAIR). What is the primary diagnosis?",
                "options": {"A": "Obturator hernia", "B": "Piriformis syndrome", "C": "Trochanteric bursitis", "D": "Meralgia paresthetica"},
                "answer_idx": "B",
            },
            {
                "question": "Numbness and tingling in the fourth and fifth digits with positive Froment's sign indicates entrapment of which nerve at Guyon's canal or cubital tunnel?",
                "options": {"A": "Median nerve", "B": "Radial nerve", "C": "Ulnar nerve", "D": "Axillary nerve"},
                "answer_idx": "C",
            },
        ]

    target_keywords = [
        "muscle", "tendon", "nerve", "pain", "radiculopathy", "plexus",
        "spine", "joint", "motor", "sensory", "reflex", "syndrome", "spasm"
    ]

    benchmark_trace = None
    if langfuse:
        try:
            benchmark_trace = langfuse.trace(
                name="medqa_offline_benchmark_100",
                metadata={"model": MODEL_ID, "target_limit": sample_limit},
            )
        except Exception as e:
            print(f"[Langfuse Trace Init Notice]: {e}")

    evaluated = 0
    correct = 0

    for item in dataset:
        if evaluated >= sample_limit:
            break

        question = item.get("question") or ""
        # Filter for relevant neuro-musculoskeletal scope
        if not any(k in question.lower() for k in target_keywords):
            continue

        # Extract 4 options
        raw_options = item.get("options")
        if isinstance(raw_options, dict):
            options = raw_options
        elif isinstance(raw_options, list):
            options = {chr(65 + i): opt for i, opt in enumerate(raw_options[:4])}
        else:
            options = {
                "A": item.get("opa", "Option A"),
                "B": item.get("opb", "Option B"),
                "C": item.get("opc", "Option C"),
                "D": item.get("opd", "Option D"),
            }

        ground_truth = str(item.get("answer_idx") or item.get("cop") or "A").strip()
        if ground_truth in ["1", "2", "3", "4"]:
            ground_truth = {"1": "A", "2": "B", "3": "C", "4": "D"}[ground_truth]

        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n"
            f"A: {options.get('A', '')}\n"
            f"B: {options.get('B', '')}\n"
            f"C: {options.get('C', '')}\n"
            f"D: {options.get('D', '')}\n\n"
            "TASK:\n"
            "Reason step-by-step through the anatomical structures, nerves, and physical signs. "
            "End your answer with exactly: 'FINAL ANSWER: <A|B|C|D>'"
        )

        span = None
        if benchmark_trace:
            try:
                span = benchmark_trace.span(
                    name=f"item_{evaluated + 1}",
                    input={"question": question, "options": options, "ground_truth": ground_truth},
                )
            except Exception:
                pass

        try:
            # Direct LLM call with correct payload
            llm_reply = call_llm(
                messages=[
                    {"role": "system", "content": "You are a clinical physician and physiotherapist solving medical licensing board questions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=220,
            )

            # Robust extraction of the option letter
            match = re.search(r"FINAL ANSWER:\s*([A-D])", llm_reply, re.IGNORECASE)
            if match:
                pred = match.group(1).upper()
            else:
                # Fallback to last mentioned capital letter or leading letter
                tail_match = re.findall(r"\b([A-D])\b", llm_reply[-40:])
                pred = tail_match[-1] if tail_match else "UNKNOWN"

        except Exception as err:
            pred = "ERROR"
            llm_reply = str(err)

        is_correct = (pred == ground_truth)
        if is_correct:
            correct += 1
        evaluated += 1

        # Direct Score Logging to Langfuse
        if benchmark_trace and langfuse:
            try:
                langfuse.score(
                    trace_id=benchmark_trace.id,
                    name="accuracy",
                    value=1.0 if is_correct else 0.0,
                    comment=f"Q{evaluated}: Pred {pred} vs Truth {ground_truth}",
                )
                if span:
                    span.end(output={"prediction": pred, "is_correct": is_correct, "excerpt": llm_reply[:200]})
            except Exception:
                pass

        current_rate = (correct / evaluated) * 100
        print(f"[{evaluated:03d}/{sample_limit}] {'PASS' if is_correct else 'FAIL'} | Pred: {pred} | Truth: {ground_truth} | Acc: {current_rate:.1f}%")

    total_accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    if benchmark_trace:
        try:
            benchmark_trace.update(output={"accuracy": total_accuracy, "correct": correct, "total": evaluated})
        except Exception:
            pass

    if langfuse:
        try:
            langfuse.flush()
            print("[Observability] Benchmark metrics successfully pushed to Langfuse.")
        except Exception:
            pass

    print("\n" + "=" * 76)
    print(f"BENCHMARK COMPLETE: {correct}/{evaluated} ({total_accuracy * 100:.2f}%)")
    print("=" * 76)

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli_mode = sys.argv[1].lower() if len(sys.argv) > 1 else "interactive"
    if cli_mode in ["benchmark", "eval", "test"]:
        item_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_offline_benchmark(sample_limit=item_limit)
    else:
        run_interactive_consultation()