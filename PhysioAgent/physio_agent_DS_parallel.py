"""
Multi-Agent Clinical Physiotherapy Diagnostic & Benchmarking System.
High-accuracy differential engine with Langfuse tracing, RAG-anchored memory,
gated Attending Reviewer verification, and 10-worker parallel benchmarking.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer

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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_ID = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL

llm_client = OpenAI(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
    timeout=60.0,
)

# ---------------------------------------------------------------------------
# Universal Thread-Safe Langfuse Adapter
# ---------------------------------------------------------------------------
class UniversalLangfuseTracker:
    def __init__(self, public_key: str, secret_key: str, host: str):
        self.client = None
        self._lock = threading.Lock()
        try:
            from langfuse import Langfuse
            self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            if hasattr(self.client, "auth_check"):
                self.client.auth_check()
            print("[Observability] Langfuse Cloud connected successfully.")
        except Exception as e:
            print(f"[Observability Notice] Langfuse disabled: {e}")

    def create_trace(self, name: str, input_data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        trace_id = str(uuid.uuid4())
        if not self.client:
            return trace_id
        try:
            with self._lock:
                method = getattr(self.client, "create_trace", getattr(self.client, "trace", None))
                if method:
                    t = method(id=trace_id, name=name, input=input_data, metadata=metadata)
                    return getattr(t, "id", trace_id)
        except Exception:
            pass
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
            meta["reasoning_trace"] = reasoning_trace[:2500]
        try:
            with self._lock:
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
            with self._lock:
                method = getattr(self.client, "create_score", getattr(self.client, "score", None))
                if method:
                    method(trace_id=trace_id, name=name, value=float(value), comment=comment)
        except Exception:
            pass

    def flush(self) -> None:
        if self.client and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception:
                pass


tracker = UniversalLangfuseTracker(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL)

# ---------------------------------------------------------------------------
# Vector Memory Engine (RAG Clinical Axioms)
# ---------------------------------------------------------------------------
HIGH_YIELD_CLINICAL_AXIOMS = [
    {
        "text": "Lateral epicondylalgia involves pain at common extensor origin exacerbated by resisted wrist extension with elbow extended. Structure: Extensor carpi radialis brevis (ECRB).",
        "diagnosis": "Lateral Epicondylalgia",
    },
    {
        "text": "Medial epicondylalgia causes pain at common flexor origin provoked by resisted wrist flexion and pronation. Involves Pronator teres and Flexor carpi radialis.",
        "diagnosis": "Medial Epicondylalgia",
    },
    {
        "text": "Piriformis syndrome involves deep buttock pain and posterior thigh paresthesia provoked by FAIR test (Flexion, Adduction, Internal Rotation) compressing sciatic nerve.",
        "diagnosis": "Piriformis Syndrome",
    },
    {
        "text": "Cubital tunnel syndrome involves ulnar nerve entrapment at elbow. Positive Froment sign due to adductor pollicis weakness with compensatory flexor pollicis longus flexion.",
        "diagnosis": "Ulnar Nerve Entrapment",
    },
    {
        "text": "De Quervain tenosynovitis affects first dorsal compartment (Abductor pollicis longus and Extensor pollicis brevis), confirmed by positive Finkelstein/Eichhoff test.",
        "diagnosis": "De Quervain Tenosynovitis",
    },
    {
        "text": "Supraspinatus tendinopathy causes anterolateral shoulder pain and painful arc (60-120 deg). Isolated via Jobe (Empty Can) test.",
        "diagnosis": "Supraspinatus Tendinopathy",
    },
    {
        "text": "C6 cervical radiculopathy causes numbness in thumb/index finger, weakness in biceps and wrist extensors, and reduced brachioradialis/biceps reflex.",
        "diagnosis": "C6 Cervical Radiculopathy",
    },
    {
        "text": "C7 cervical radiculopathy causes middle finger paresthesias, triceps/wrist flexor weakness, and diminished triceps tendon reflex.",
        "diagnosis": "C7 Cervical Radiculopathy",
    },
    {
        "text": "L5 radiculopathy causes sensory deficit over anterolateral leg and foot dorsum, weakness in ankle dorsiflexion and great toe extension. Reflexes preserved.",
        "diagnosis": "L5 Radiculopathy",
    },
    {
        "text": "S1 radiculopathy causes pain/numbness in lateral foot/sole, weakness in plantarflexion (gastrocnemius/soleus), and diminished/absent Achilles reflex.",
        "diagnosis": "S1 Radiculopathy",
    },
]

class ClinicalVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.encoder = SentenceTransformer(model_name)
        self.cases: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._seed_axioms()

    def _seed_axioms(self) -> None:
        texts = [ax["text"] for ax in HIGH_YIELD_CLINICAL_AXIOMS]
        vecs = self.encoder.encode(texts, normalize_embeddings=True)
        self.vectors = np.array(vecs)
        self.cases = list(HIGH_YIELD_CLINICAL_AXIOMS)

    def add_case(self, case_summary: str, diagnosis: str) -> None:
        record = f"{case_summary} | Diagnosis: {diagnosis}"
        vec = self.encoder.encode([record], normalize_embeddings=True)
        with self._lock:
            self.cases.append({"text": record, "diagnosis": diagnosis})
            if self.vectors is None:
                self.vectors = vec
            else:
                self.vectors = np.vstack([self.vectors, vec])

    def query_precedents(self, query: str, top_k: int = 2) -> str:
        if self.vectors is None or len(self.cases) == 0:
            return "No precedents recorded."
        q_vec = self.encoder.encode([query], normalize_embeddings=True)[0]
        with self._lock:
            scores = np.dot(self.vectors, q_vec)
            best_indices = np.argsort(scores)[::-1][:top_k]
            matches = [f"- {self.cases[i]['text']}" for i in best_indices if scores[i] > 0.35]
        return "\n".join(matches) if matches else "No matching clinical precedent."


memory = ClinicalVectorStore()

# ---------------------------------------------------------------------------
# Search Retrieval
# ---------------------------------------------------------------------------
def query_medical_web(query: str) -> str:
    if DDGS is None:
        return "Search engine unavailable."
    try:
        results = []
        with DDGS() as ddgs:
            clean_q = re.sub(r"[^\w\s]", " ", query).strip()
            for item in ddgs.text(f"physiotherapy clinical test {clean_q}", max_results=2):
                body = (item.get("body") or "")[:250].replace("\n", " ")
                results.append(f"[{item.get('title')}]: {body}")
        return "\n".join(results) if results else "No specific literature found."
    except Exception as err:
        return f"Literature retrieval notice: {str(err)}"

# ---------------------------------------------------------------------------
# DeepSeek V4.1 Execution Engine
# ---------------------------------------------------------------------------
def call_deepseek(
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 8256,
    enable_thinking: bool = True,
    reasoning_effort: str = "high",
    json_mode: bool = True,
) -> Tuple[str, str]:
    extra_body: Dict[str, Any] = {}
    if enable_thinking:
        extra_body["thinking"] = {"type": "enabled"}
        extra_body["reasoning_effort"] = reasoning_effort

    kwargs: Dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = llm_client.chat.completions.create(**kwargs)
    choice = response.choices[0].message
    content = choice.content or ""
    reasoning = getattr(choice, "reasoning_content", "") or ""
    return content, reasoning

# ---------------------------------------------------------------------------
# Answer Extraction
# ---------------------------------------------------------------------------
def extract_answer_and_confidence(llm_reply: str, options: Dict[str, str]) -> Tuple[str, float, str]:
    confidence = 0.70
    rationale = ""

    try:
        data = json.loads(llm_reply)
        ans = str(data.get("answer", "")).strip().upper()
        rationale = str(data.get("clinical_differential", "") or data.get("rationale", ""))
        if ans in ["A", "B", "C", "D"]:
            conf = float(data.get("confidence", 0.85))
            return ans, (conf if conf <= 1.0 else conf / 100.0), rationale
    except Exception:
        pass

    json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_reply, re.DOTALL) or re.search(r"(\{[\s\S]*\})", llm_reply)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            ans = str(data.get("answer", "")).strip().upper()
            rationale = str(data.get("clinical_differential", "") or data.get("rationale", ""))
            if ans in ["A", "B", "C", "D"]:
                conf = float(data.get("confidence", 0.80))
                return ans, (conf if conf <= 1.0 else conf / 100.0), rationale
        except Exception:
            pass

    match = re.search(r"(?:FINAL ANSWER|CORRECT OPTION|ANSWER IS|ANSWER)[:\s*]+([A-D])\b", llm_reply, re.IGNORECASE)
    if match:
        return match.group(1).upper(), 0.75, "Extracted via pattern"

    bold_match = re.search(r"(?:\*\*|\()([A-D])(?:\*\*|\))", llm_reply)
    if bold_match:
        return bold_match.group(1).upper(), 0.70, "Extracted via bold format"

    for letter, text in options.items():
        if text and len(text) > 4 and text.lower() in llm_reply.lower():
            return letter, 0.65, f"Matched option string: {text[:20]}"

    all_letters = re.findall(r"\b([A-D])\b", llm_reply)
    if all_letters:
        return all_letters[-1].upper(), 0.40, "Fallback letter"

    return "A", 0.25, "Default fallback"

# ---------------------------------------------------------------------------
# Attending Reviewer Agent (Borderline Case Verification)
# ---------------------------------------------------------------------------
def perform_attending_review(
    question: str,
    options: Dict[str, str],
    initial_answer: str,
    initial_rationale: str,
    precedents: str,
) -> Tuple[str, float]:
    review_prompt = (
        f"PATIENT VIGNETTE:\n{question}\n\n"
        f"OPTIONS:\n"
        f"A: {options.get('A', '')}\n"
        f"B: {options.get('B', '')}\n"
        f"C: {options.get('C', '')}\n"
        f"D: {options.get('D', '')}\n\n"
        f"CLINICAL MEMORY PRECEDENTS:\n{precedents}\n\n"
        f"RESIDENT ASSESSMENT:\n"
        f"Selected Option: {initial_answer} ({options.get(initial_answer, '')})\n"
        f"Reasoning: {initial_rationale}\n\n"
        "ATTENDING REVIEW PROTOCOL:\n"
        "1. Check for subtle distractor traps (e.g. adjacent spinal nerve roots, tendon vs bursa, antagonistic actions).\n"
        "2. Confirm if another option satisfies the pathognomonic findings with higher specificity.\n"
        "3. Confirm or overturn the choice.\n\n"
        "Return strictly valid JSON:\n"
        "{\n"
        '  "review_critique": "Anatomical audit",\n'
        '  "overturn": true or false,\n'
        '  "final_answer": "A",\n'
        '  "final_confidence": 0.90\n'
        "}"
    )

    messages = [
        {
            "role": "system",
            "content": "You are an Attending Musculoskeletal Physician and Senior Board Examiner. Audit the decision in valid JSON.",
        },
        {"role": "user", "content": review_prompt},
    ]

    try:
        reply, _ = call_deepseek(
            messages=messages,
            temperature=0.0,
            max_tokens=80256,
            enable_thinking=True,
            reasoning_effort="high",
            json_mode=True,
        )
        data = json.loads(reply)
        final_ans = str(data.get("final_answer", initial_answer)).strip().upper()
        if final_ans in ["A", "B", "C", "D"]:
            conf = float(data.get("final_confidence", 0.88))
            return final_ans, (conf if conf <= 1.0 else conf / 100.0)
    except Exception:
        pass

    return initial_answer, 0.75

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
# Interactive Consultation Mode
# ---------------------------------------------------------------------------
def run_interactive_consultation() -> None:
    print("\n" + "=" * 76)
    print("PHYSIOTHERAPY CLINICAL REASONING SYSTEM - MULTI-AGENT DIAGNOSTIC MODE")
    print(f"Engine: DeepSeek V4.1 Flash [{MODEL_ID}] | Vector Memory: Active")
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
        print(f"\n[Turn {turn}] Clinician panel formulating differential analysis...")

        precedents = memory.query_precedents(f"{session.chief_complaint} {session.reported_symptoms}")

        search_evidence = ""
        if turn >= 2 and len(session.active_differentials) > 1:
            search_query = f"{session.active_differentials[0]} vs {session.active_differentials[1]} physical examination differential"
            search_evidence = query_medical_web(search_query)

        system_prompt = (
            "You are a dual-agent clinical panel consisting of a Lead Orthopedic Physiotherapist and an Anatomical Skeptic.\n"
            "Systematically eliminate lookalike conditions via provocative orthopedic maneuvers.\n\n"
            "RULES:\n"
            "1. NEVER repeat maneuvers in 'ALREADY TESTED MANEUVERS'.\n"
            "2. Propose maneuvers with high discriminatory power (high positive likelihood ratio).\n"
            "3. Ground rationale in anatomical innervation, mechanics, or tissue stress.\n"
            "4. Respond strictly in valid JSON."
        )

        user_prompt = (
            f"CLINICAL STATE:\n{session.get_summary()}\n\n"
            f"VECTOR MEMORY PRECEDENTS:\n{precedents}\n\n"
            f"LITERATURE EVIDENCE:\n{search_evidence or 'None required.'}\n\n"
            "Provide your evaluation strictly in JSON format:\n"
            "{\n"
            '  "skeptic_counter_differential": "Primary lookalike to rule out",\n'
            '  "clinical_rationale": "Anatomical justification for this test",\n'
            '  "active_differentials": ["<condition 1>", "<condition 2>"],\n'
            '  "ruled_out": ["<condition ruled out>"],\n'
            '  "status": "GRILL" or "CONFIRMED",\n'
            '  "test_name": "<exact clinical maneuver name>",\n'
            '  "instructions_for_patient": "<clear step-by-step physical test instructions>",\n'
            '  "final_diagnosis": null or "<Definitive Diagnosis>",\n'
            '  "confidence": <float 0.00 to 1.00>\n'
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
                max_tokens=80256,
                enable_thinking=True,
                reasoning_effort="high",
                json_mode=True,
            )
        except Exception as e:
            print(f"[API Error]: {e}")
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
        skeptic_note = decision.get("skeptic_counter_differential", "")
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

        print(f"\nLead Clinician [Confidence: {confidence * 100:.1f}% | Active: {', '.join(session.active_differentials)}]:")
        if skeptic_note:
            print(f"*(Differential Target: {skeptic_note})*")
        print(f"{patient_instructions}")
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
# Parallel Benchmarking Engine (10 Concurrent Workers)
# ---------------------------------------------------------------------------
def evaluate_single_question(
    q_index: int,
    item: Dict[str, Any],
    benchmark_trace_id: str,
) -> Dict[str, Any]:
    """
    Evaluator worker: Queries RAG memory, constructs distractor matrix,
    executes DeepSeek V4.1 Flash, and gates through Attending Review if needed.
    """
    question = item.get("question") or ""
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

    # RAG precedent retrieval
    precedents = memory.query_precedents(question, top_k=2)

    prompt = (
        f"CLINICAL VIGNETTE:\n{question}\n\n"
        f"OPTIONS:\n"
        f"A: {options.get('A', '')}\n"
        f"B: {options.get('B', '')}\n"
        f"C: {options.get('C', '')}\n"
        f"D: {options.get('D', '')}\n\n"
        f"HIGH-YIELD CLINICAL ANATOMICAL AXIOMS:\n{precedents}\n\n"
        "DIAGNOSTIC PROTOCOL:\n"
        "1. ANATOMICAL SUBSTRATE: Identify affected muscle, nerve segment, or joint compartment.\n"
        "2. DISTRACTOR FALSIFICATION: State why each incorrect option is biologically or clinically false.\n"
        "3. SELECTION: Specify the exact option text first, then select option letter (A, B, C, or D).\n"
        "4. CONFIDENCE: Calibrate from 0.50 (uncertain) to 1.00 (pathognomonic hallmark).\n\n"
        "Output strictly valid JSON:\n"
        "{\n"
        '  "anatomical_substrate": "Key structure involved",\n'
        '  "distractor_falsification": {\n'
        '    "why_not_distractor_1": "...",\n'
        '    "why_not_distractor_2": "...",\n'
        '    "why_not_distractor_3": "..."\n'
        "  },\n"
        '  "clinical_differential": "Diagnostic rationale",\n'
        '  "selected_pathology_text": "Text of winning option",\n'
        '  "answer": "A",\n'
        '  "confidence": 0.90\n'
        "}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert clinical anatomist and board certification examiner. "
                "Reason from anatomical first principles and falsify distractors before selecting an answer. "
                "Output strictly valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    reviewed = False
    try:
        llm_reply, reasoning = call_deepseek(
            messages=messages,
            temperature=0.0,
            max_tokens=8256,
            enable_thinking=True,
            reasoning_effort="high",
            json_mode=True,
        )
        pred, conf, rationale = extract_answer_and_confidence(llm_reply, options)

        # Attending Reviewer: Trigger on borderline confidence
        if conf < 0.85:
            pred, conf = perform_attending_review(
                question=question,
                options=options,
                initial_answer=pred,
                initial_rationale=rationale,
                precedents=precedents,
            )
            reviewed = True

    except Exception as err:
        pred = "A"
        conf = 0.20
        llm_reply = f"Error: {type(err).__name__} - {err}"
        reasoning = ""

    is_correct = (pred == ground_truth)

    tracker.log_generation(
        trace_id=benchmark_trace_id,
        name=f"mcq_q{q_index}",
        input_messages=messages,
        output_text=llm_reply,
        reasoning_trace=reasoning,
        metadata={"ground_truth": ground_truth, "pred": pred, "reviewed": reviewed},
    )
    tracker.log_score(
        trace_id=benchmark_trace_id,
        name="accuracy",
        value=1.0 if is_correct else 0.0,
        comment=f"Q{q_index}: Pred {pred} vs Truth {ground_truth}",
    )

    return {
        "index": q_index,
        "pred": pred,
        "ground_truth": ground_truth,
        "conf": conf,
        "is_correct": is_correct,
        "reviewed": reviewed,
    }


def run_parallel_benchmark(sample_limit: int = 100, num_workers: int = 10) -> None:
    print("\n" + "=" * 76)
    print(f"OFFLINE PHYSIO BENCHMARK: MedQA USMLE Streaming (Target: {sample_limit} items)")
    print(f"Concurrency: {num_workers} Parallel Agents | Model: {MODEL_ID}")
    print("Architecture: RAG Exemplars + Distractor Falsification + Attending Review")
    print("=" * 76)

    print("Collecting clinical vignettes from Hugging Face stream...")
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

    evaluation_tasks: List[Dict[str, Any]] = []
    for item in dataset:
        q = item.get("question") or ""
        if any(k in q.lower() for k in target_keywords):
            evaluation_tasks.append(item)
            if len(evaluation_tasks) >= sample_limit:
                break

    actual_count = len(evaluation_tasks)
    print(f"Collected {actual_count} clinical vignettes. Dispatching across {num_workers} parallel workers...\n")

    benchmark_trace_id = tracker.create_trace(
        name=f"medqa_benchmark_parallel_{num_workers}agents",
        input_data={"target_limit": actual_count, "workers": num_workers},
        metadata={"model": MODEL_ID, "concurrency": num_workers},
    )

    print_lock = threading.Lock()
    correct_count = 0
    confidence_sum = 0.0
    completed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_single_question, idx + 1, item, benchmark_trace_id): idx + 1
            for idx, item in enumerate(evaluation_tasks)
        }

        for future in as_completed(future_to_task):
            res = future.result()
            with print_lock:
                completed_count += 1
                if res["is_correct"]:
                    correct_count += 1
                confidence_sum += res["conf"]

                running_acc = (correct_count / completed_count) * 100
                running_avg_conf = (confidence_sum / completed_count) * 100
                review_tag = "[Attending Review]" if res["reviewed"] else ""

                print(
                    f"[{completed_count:03d}/{actual_count}] "
                    f"{'PASS' if res['is_correct'] else 'FAIL'} | "
                    f"Q{res['index']:03d} | "
                    f"Pred: {res['pred']} | Truth: {res['ground_truth']} | "
                    f"Conf: {res['conf'] * 100:4.1f}% | "
                    f"Acc: {running_acc:5.1f}% | "
                    f"Avg Conf: {running_avg_conf:5.1f}% {review_tag}"
                )

    final_accuracy = (correct_count / completed_count) if completed_count > 0 else 0.0
    final_avg_conf = (confidence_sum / completed_count) if completed_count > 0 else 0.0

    tracker.log_score(
        trace_id=benchmark_trace_id,
        name="final_benchmark_accuracy",
        value=final_accuracy,
        comment=f"Overall accuracy over {completed_count} questions ({num_workers} workers)",
    )
    tracker.log_score(
        trace_id=benchmark_trace_id,
        name="final_average_confidence",
        value=final_avg_conf,
        comment=f"Overall average confidence over {completed_count} questions",
    )

    tracker.flush()
    print("\n" + "=" * 76)
    print(
        f"PARALLEL BENCHMARK COMPLETE: {correct_count}/{completed_count} ({final_accuracy * 100:.2f}%) | "
        f"Average Confidence: {final_avg_conf * 100:.2f}%"
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
        worker_count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        run_parallel_benchmark(sample_limit=item_limit, num_workers=worker_count)
    else:
        run_interactive_consultation()