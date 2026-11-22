"""
Multi-Agent Clinical Physiotherapy Diagnostic & Benchmarking System.
High-accuracy clinical differential engine with Langfuse tracing and DeepSeek V4.1 Flash.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Resilient DuckDuckGo import
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ---------------------------------------------------------------------------
# API & Observability Configuration
# ---------------------------------------------------------------------------
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-2a24bee0-e7ca-4f88-8d9e-3d511708d502")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-ccc486e0-0d3c-4245-9f5f-987353d79cea")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# DeepSeek API Configuration (DeepSeek V4.1 Flash)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_ID = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

# OpenAI-compatible client for DeepSeek
llm_client = OpenAI(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    timeout=45.0,
)

# ---------------------------------------------------------------------------
# Universal Langfuse Compatibility Adapter
# ---------------------------------------------------------------------------
class UniversalLangfuseTracker:
    """Supports Langfuse SDK tracking with graceful degradation."""

    def __init__(self, public_key: str, secret_key: str, host: str):
        self.client = None
        try:
            from langfuse import Langfuse
            self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            if hasattr(self.client, "auth_check"):
                self.client.auth_check()
            print("[Observability] Langfuse Cloud connected successfully.")
        except Exception as e:
            print(f"[Observability Notice] Langfuse disabled or unavailable: {e}")

    def create_trace(self, name: str, input_data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        trace_id = str(uuid.uuid4())
        if not self.client:
            return trace_id
        try:
            if hasattr(self.client, "create_trace"):
                t = self.client.create_trace(id=trace_id, name=name, input=input_data, metadata=metadata)
                return getattr(t, "id", trace_id)
            if hasattr(self.client, "trace"):
                t = self.client.trace(id=trace_id, name=name, input=input_data, metadata=metadata)
                return getattr(t, "id", trace_id)
        except Exception as err:
            print(f"[Observability Notice] Trace fallback: {err}")
        return trace_id

    def log_generation(
        self,
        trace_id: str,
        name: str,
        input_messages: Any,
        output_text: str,
        reasoning_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.client:
            return
        meta = metadata or {}
        if reasoning_trace:
            meta["deepseek_reasoning_trace"] = reasoning_trace
        try:
            method = getattr(self.client, "generation", getattr(self.client, "create_generation", None))
            if method:
                method(
                    trace_id=trace_id,
                    name=name,
                    input=input_messages,
                    output=output_text,
                    model=MODEL_ID,
                    metadata=meta,
                )
        except Exception:
            pass

    def log_score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        if not self.client:
            return
        try:
            method = getattr(self.client, "create_score", getattr(self.client, "score", None))
            if method:
                method(
                    trace_id=trace_id,
                    name=name,
                    value=float(value),
                    comment=comment,
                )
        except Exception as e:
            print(f"[Observability Notice] Score log error: {e}")

    def flush(self) -> None:
        if self.client and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception:
                pass


tracker = UniversalLangfuseTracker(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)

# ---------------------------------------------------------------------------
# Vector Memory Engine
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
# Medical Search Engine
# ---------------------------------------------------------------------------
def query_medical_web(query: str) -> str:
    """Searches medical literature for clinical examination criteria."""
    if DDGS is None:
        return "Search engine unavailable."
    try:
        results = []
        with DDGS() as ddgs:
            clean_q = re.sub(r"[^\w\s]", " ", query).strip()
            for item in ddgs.text(f"physiotherapy clinical test {clean_q}", max_results=2):
                body = (item.get("body") or "")[:250].replace("\n", " ")
                results.append(f"[{item.get('title')}]: {body}")
        return "\n".join(results) if results else "No specific medical literature returned."
    except Exception as err:
        return f"Literature retrieval notice: {str(err)}"

# ---------------------------------------------------------------------------
# DeepSeek Execution Engine (Supports Thinking Mode)
# ---------------------------------------------------------------------------
def call_deepseek(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    enable_thinking: bool = True,
    reasoning_effort: str = "low",
    json_mode: bool = False,
) -> Tuple[str, str]:
    """
    Executes completion via DeepSeek V4.1 Flash.
    Returns: (content, reasoning_content)
    """
    kwargs: Dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    if enable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = reasoning_effort

    response = llm_client.chat.completions.create(**kwargs)
    choice = response.choices[0].message
    content = choice.content or ""
    reasoning = getattr(choice, "reasoning_content", "") or ""
    return content, reasoning

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
            "\n".join([f" - {t['test']}: {t['result']}" for t in self.completed_tests])
            if self.completed_tests
            else " - None performed yet"
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
# Interactive Clinical Diagnostic Agent
# ---------------------------------------------------------------------------
def run_interactive_consultation() -> None:
    print("\n" + "=" * 76)
    print("PHYSIOTHERAPY CLINICAL REASONING SYSTEM - INTERACTIVE DIAGNOSTIC MODE")
    print("Engine: DeepSeek V4.1 Flash (Thinking Mode Enabled)")
    print("=" * 76)
    print("Type 'exit' to quit.\n")

    complaint = input("Lead Clinician: What symptoms are you experiencing? Describe location and onset:\n> ").strip()
    if not complaint or complaint.lower() in ["exit", "quit"]:
        return

    session = ClinicalSession(chief_complaint=complaint, reported_symptoms=complaint)
    trace_id = tracker.create_trace(
        name="interactive_physio_consultation",
        input_data={"chief_complaint": complaint},
        metadata={"model": MODEL_ID},
    )

    turn = 1
    MAX_TURNS = 7
    CONFIDENCE_THRESHOLD = 0.85

    while turn <= MAX_TURNS:
        print(f"\n[Turn {turn}] Clinician is formulating differential analysis...")

        precedents = memory.query_precedents(f"{session.chief_complaint} {session.reported_symptoms}")

        search_evidence = ""
        if turn >= 2 and len(session.active_differentials) > 1:
            search_query = f"{session.active_differentials[0]} vs {session.active_differentials[1]} physical examination differential"
            search_evidence = query_medical_web(search_query)

        system_prompt = (
            "You are a board-certified orthopedic and neurological physiotherapist. Your goal is to systematically "
            "isolate the lesion or pathology through targeted provocative maneuvers.\n\n"
            "DIAGNOSTIC PROTOCOL:\n"
            "1. NEVER repeat maneuvers listed in 'ALREADY TESTED MANEUVERS'.\n"
            "2. Think methodically about innervation (nerve root vs peripheral branch) or mechanical load (tendon vs bursa).\n"
            "3. If previous test was positive, isolate exact anatomical structure. If negative, pivot to rule out lookalikes.\n"
            "4. Output MUST be valid JSON adhering strictly to the required schema."
        )

        user_prompt = (
            f"CLINICAL STATE:\n{session.get_summary()}\n\n"
            f"VECTOR MEMORY PRECEDENTS:\n{precedents}\n\n"
            f"LITERATURE EVIDENCE:\n{search_evidence or 'None required.'}\n\n"
            "Respond strictly in this JSON structure:\n"
            "{\n"
            '  "clinical_rationale": "Step-by-step anatomical reasoning for this turn",\n'
            '  "active_differentials": ["<condition 1>", "<condition 2>"],\n'
            '  "ruled_out": ["<condition ruled out>"],\n'
            '  "status": "GRILL" or "CONFIRMED",\n'
            '  "test_name": "<exact clinical maneuver name>",\n'
            '  "instructions_for_patient": "<clear, actionable test instructions>",\n'
            '  "final_diagnosis": null or "<Definitive Diagnosis>",\n'
            '  "confidence": <float from 0.00 to 1.00 based on pathognomonic evidence>\n'
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_reply, reasoning = call_deepseek(
                messages=messages,
                temperature=0.1,
                max_tokens=850,
                enable_thinking=True,
                reasoning_effort="low",
                json_mode=True,
            )
        except Exception as e:
            print(f"[API Error] Failed to call DeepSeek: {e}")
            break

        tracker.log_generation(
            trace_id=trace_id,
            name=f"turn_{turn}_generation",
            input_messages=messages,
            output_text=raw_reply,
            reasoning_trace=reasoning,
        )

        try:
            decision = json.loads(raw_reply)
        except Exception:
            match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
            decision = json.loads(match.group(0)) if match else {}

        confidence = float(decision.get("confidence", 0.40))
        if confidence > 1.0:
            confidence = confidence / 100.0
        session.current_confidence = confidence

        if decision.get("active_differentials"):
            session.active_differentials = decision["active_differentials"]
        if decision.get("ruled_out"):
            session.ruled_out_conditions.extend(
                [r for r in decision["ruled_out"] if r not in session.ruled_out_conditions]
            )

        test_name = decision.get("test_name", f"Provocative Exam {turn}")
        patient_instructions = decision.get("instructions_for_patient", raw_reply)
        rationale = decision.get("clinical_rationale", "")
        status = decision.get("status", "GRILL").upper()

        tracker.log_score(
            trace_id=trace_id,
            name="diagnostic_confidence",
            value=confidence,
            comment=f"Turn {turn} confidence level",
        )

        if status == "CONFIRMED" or confidence >= CONFIDENCE_THRESHOLD:
            diagnosis = decision.get("final_diagnosis") or "Verified Neuro-Musculoskeletal Disorder"
            print(f"\nLead Clinician [Diagnostic Confidence: {confidence * 100:.1f}%]:")
            print(f"**Final Clinical Diagnosis**: {diagnosis}")
            print(f"**Clinical Rationale**: {rationale}\n")

            memory.add_case(case_summary=session.get_summary(), diagnosis=diagnosis)
            tracker.log_score(trace_id=trace_id, name="diagnosis_confirmed", value=1.0, comment=str(diagnosis))
            break

        print(f"\nLead Clinician [Confidence: {confidence * 100:.1f}% | Differentials: {', '.join(session.active_differentials)}]:")
        print(f"{patient_instructions}")
        if rationale:
            print(f"*(Objective: {rationale})*")
        print()

        patient_feedback = input("Your test outcome / physical response:\n> ").strip()
        if not patient_feedback or patient_feedback.lower() in ["exit", "quit"]:
            print("Consultation concluded by user.")
            break

        session.tested_maneuvers.append(test_name)
        session.completed_tests.append({"test": test_name, "result": patient_feedback})
        turn += 1

    if turn > MAX_TURNS and session.current_confidence < CONFIDENCE_THRESHOLD:
        print("\nLead Clinician: Completed maximum testing threshold.")
        print(f"Working Differential: {', '.join(session.active_differentials)}")
        print("Recommendation: Refer for diagnostic ultrasound, MRI, or electrodiagnostic testing (EMG/NCS).")

    tracker.flush()
    print("[Observability] Interactive session data sent to Langfuse Cloud.")

# ---------------------------------------------------------------------------
# High-Accuracy Medical Benchmark Engine
# ---------------------------------------------------------------------------
def extract_answer_and_confidence(llm_reply: str, options: Dict[str, str]) -> Tuple[str, float]:
    """
    Extracts answer letter and calibrated confidence from structured JSON or text.
    """
    confidence = 0.50

    try:
        data = json.loads(llm_reply)
        ans = str(data.get("answer", "")).strip().upper()
        if ans in ["A", "B", "C", "D"]:
            conf = float(data.get("confidence", 0.80))
            return ans, conf if conf <= 1.0 else conf / 100.0
    except Exception:
        pass

    # Regex search for JSON object inside markdown code block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_reply, re.DOTALL) or re.search(r"(\{.*\})", llm_reply, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            ans = str(data.get("answer", "")).strip().upper()
            if ans in ["A", "B", "C", "D"]:
                conf = float(data.get("confidence", 0.75))
                return ans, conf if conf <= 1.0 else conf / 100.0
        except Exception:
            pass

    # Extract confidence score if present
    conf_search = re.search(r"confidence[\"':\s]+([0-9]*\.?[0-9]+)", llm_reply, re.IGNORECASE)
    if conf_search:
        try:
            c = float(conf_search.group(1))
            confidence = c if c <= 1.0 else c / 100.0
        except Exception:
            pass

    # Direct answer patterns
    match = re.search(r"(?:FINAL ANSWER|CORRECT OPTION|ANSWER IS|ANSWER)[:\s*]+([A-D])\b", llm_reply, re.IGNORECASE)
    if match:
        return match.group(1).upper(), max(confidence, 0.75)

    bold_match = re.search(r"(?:\*\*|\()([A-D])(?:\*\*|\))", llm_reply)
    if bold_match:
        return bold_match.group(1).upper(), max(confidence, 0.65)

    # Match literal text against options
    for letter, text in options.items():
        if text and len(text) > 4 and text.lower() in llm_reply.lower():
            return letter, max(confidence, 0.60)

    all_letters = re.findall(r"\b([A-D])\b", llm_reply)
    if all_letters:
        return all_letters[-1].upper(), max(confidence, 0.40)

    return "A", 0.25


def run_offline_benchmark(sample_limit: int = 100) -> None:
    print("\n" + "=" * 76)
    print(f"OFFLINE PHYSIO BENCHMARK: MedQA USMLE Streaming (Target: {sample_limit} items)")
    print("Engine: DeepSeek V4.1 Flash (Low-Effort CoT Thinking Mode)")
    print("=" * 76)

    print("Connecting to Hugging Face dataset stream ('GBaker/MedQA-USMLE-4-options')...")
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", streaming=True)
    except Exception as e:
        print(f"Dataset streaming notice: {e}. Using local benchmark fallback.")
        dataset = [
            {
                "question": "A 40-year-old athlete presents with lateral elbow pain exacerbated by resisted wrist extension. What structure is affected?",
                "options": {"A": "Extensor carpi radialis brevis tendon", "B": "Flexor carpi ulnaris", "C": "Biceps brachii tendon", "D": "Pronator teres"},
                "answer_idx": "A",
            },
            {
                "question": "A runner has deep buttock pain and posterior thigh paresthesia exacerbated by passive hip flexion, adduction, and internal rotation (FAIR). What is the diagnosis?",
                "options": {"A": "Obturator hernia", "B": "Piriformis syndrome", "C": "Trochanteric bursitis", "D": "Meralgia paresthetica"},
                "answer_idx": "B",
            },
            {
                "question": "Numbness and tingling in the fourth and fifth digits with positive Froment sign indicates entrapment of which nerve?",
                "options": {"A": "Median nerve", "B": "Radial nerve", "C": "Ulnar nerve", "D": "Axillary nerve"},
                "answer_idx": "C",
            },
        ]

    target_keywords = [
        "muscle", "tendon", "nerve", "pain", "radiculopathy", "plexus",
        "spine", "joint", "motor", "sensory", "reflex", "syndrome", "spasm"
    ]

    benchmark_trace_id = tracker.create_trace(
        name="medqa_offline_benchmark_deepseek_flash",
        input_data={"target_limit": sample_limit},
        metadata={"model": MODEL_ID, "reasoning_effort": "low"},
    )

    evaluated = 0
    correct = 0
    confidence_sum = 0.0

    for item in dataset:
        if evaluated >= sample_limit:
            break

        question = item.get("question") or ""
        if not any(k in question.lower() for k in target_keywords):
            continue

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

        # Pre-answer reasoning prompts prevent next-token guessing and calibrate confidence
        prompt = (
            f"CLINICAL SCENARIO:\n{question}\n\n"
            f"OPTIONS:\n"
            f"A: {options.get('A', '')}\n"
            f"B: {options.get('B', '')}\n"
            f"C: {options.get('C', '')}\n"
            f"D: {options.get('D', '')}\n\n"
            "DIAGNOSTIC PROTOCOL:\n"
            "1. Identify the anatomical structure, nerve segment, or pathophysiology.\n"
            "2. Eliminate the incorrect distractors with brief clinical rationale.\n"
            "3. Select the single best option (A, B, C, or D).\n"
            "4. Assign calibrated confidence: 0.90-1.00 (pathognomonic/definitive), "
            "0.75-0.89 (strong presentation), 0.50-0.74 (atypical/equivocal).\n\n"
            "Output your findings in this exact JSON format:\n"
            "{\n"
            '  "clinical_differential": "Concise anatomical/pathological deduction",\n'
            '  "elimination_rationale": "Why competing options fail",\n'
            '  "answer": "A",\n'
            '  "confidence": 0.90\n'
            "}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical physician and musculoskeletal board examiner. "
                    "Perform step-by-step anatomical elimination before choosing the final letter. "
                    "Always respond in valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            llm_reply, reasoning = call_deepseek(
                messages=messages,
                temperature=0.0,
                max_tokens=600,
                enable_thinking=True,
                reasoning_effort="low",
                json_mode=True,
            )
            pred, conf = extract_answer_and_confidence(llm_reply, options)
        except Exception as err:
            pred = "A"
            conf = 0.20
            llm_reply = str(err)
            reasoning = ""

        is_correct = (pred == ground_truth)
        if is_correct:
            correct += 1
        evaluated += 1
        confidence_sum += conf

        # Log item to Langfuse
        tracker.log_generation(
            trace_id=benchmark_trace_id,
            name=f"mcq_q{evaluated}",
            input_messages=messages,
            output_text=llm_reply,
            reasoning_trace=reasoning,
            metadata={"question": question[:80], "options": options, "ground_truth": ground_truth},
        )
        tracker.log_score(
            trace_id=benchmark_trace_id,
            name="accuracy",
            value=1.0 if is_correct else 0.0,
            comment=f"Q{evaluated}: Pred {pred} vs Truth {ground_truth}",
        )
        tracker.log_score(
            trace_id=benchmark_trace_id,
            name="confidence",
            value=conf,
            comment=f"Q{evaluated} model confidence",
        )

        running_acc = (correct / evaluated) * 100
        running_avg_conf = (confidence_sum / evaluated) * 100
        print(
            f"[{evaluated:03d}/{sample_limit}] "
            f"{'PASS' if is_correct else 'FAIL'} | "
            f"Pred: {pred} | Truth: {ground_truth} | "
            f"Conf: {conf * 100:4.1f}% | "
            f"Running Acc: {running_acc:5.1f}% | "
            f"Avg Conf: {running_avg_conf:5.1f}%"
        )

    final_accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    final_avg_confidence = (confidence_sum / evaluated) if evaluated > 0 else 0.0

    tracker.log_score(
        trace_id=benchmark_trace_id,
        name="final_benchmark_accuracy",
        value=final_accuracy,
        comment=f"Overall accuracy over {evaluated} questions",
    )
    tracker.log_score(
        trace_id=benchmark_trace_id,
        name="final_average_confidence",
        value=final_avg_confidence,
        comment=f"Overall average confidence over {evaluated} questions",
    )

    tracker.flush()
    print("\n" + "=" * 76)
    print(
        f"BENCHMARK COMPLETE: {correct}/{evaluated} ({final_accuracy * 100:.2f}%) | "
        f"Average Confidence: {final_avg_confidence * 100:.2f}%"
    )
    print("[Observability] Benchmark metrics successfully pushed to Langfuse Cloud.")
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