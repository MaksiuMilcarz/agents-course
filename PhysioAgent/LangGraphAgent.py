import os
import json
import operator
from typing import Annotated, Dict, List, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Langfuse v3+ integration
from langfuse.langchain import CallbackHandler
from langfuse import get_client

# Terminal Visualization
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.json import JSON

console = Console()

# ---------------------------------------------------------------------------
# 1. Environment & Observability Configuration
# ---------------------------------------------------------------------------
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-2a24bee0-e7ca-4f88-8d9e-3d511708d502"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-ccc486e0-0d3c-4245-9f5f-987353d79cea"
os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"

OMLX_API_KEY = "sk-omlx-gWACPPkjsP67zAEgJ3GReJ6q"
OMLX_BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-mlx-4Bit"

langfuse_client = get_client()
langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    model=MODEL_ID,
    api_key=OMLX_API_KEY,
    base_url=OMLX_BASE_URL,
    temperature=0.0,
    callbacks=[langfuse_handler]
)

# ---------------------------------------------------------------------------
# 2. Schema and State Definitions
# ---------------------------------------------------------------------------
class CustomerRecord(BaseModel):
    customer_id: str
    age: int = Field(ge=18, le=100)
    monthly_charges: float = Field(gt=0.0, le=1000.0)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    status: str = "valid"

class PipelineState(TypedDict):
    raw_batch: List[Dict]
    clean_records: List[Dict]
    corrupted_records: List[Dict]
    repaired_records: List[Dict]
    rejection_log: Annotated[List[str], operator.add]
    audit_report: str

# ---------------------------------------------------------------------------
# 3. Graph Nodes
# ---------------------------------------------------------------------------
def deterministic_validator(state: PipelineState) -> Dict:
    clean = []
    corrupted = []
    logs = []

    for item in state["raw_batch"]:
        cid = item.get("customer_id", "UNKNOWN")
        errors = []

        try:
            age = int(item.get("age"))
            if not (18 <= age <= 100):
                errors.append(f"Age {age} out of range [18, 100]")
        except (ValueError, TypeError):
            errors.append("Age missing or not integer")

        try:
            charges = float(item.get("monthly_charges"))
            if not (0.0 < charges <= 1000.0):
                errors.append(f"Charges {charges} out of bounds")
        except (ValueError, TypeError):
            errors.append("Charges missing or not float")

        contract = str(item.get("contract_type", "")).strip().lower()
        if contract not in ["month-to-month", "one-year", "two-year"]:
            errors.append(f"Invalid contract '{contract}'")

        if not errors:
            clean.append({
                "customer_id": cid,
                "age": int(item["age"]),
                "monthly_charges": float(item["monthly_charges"]),
                "contract_type": contract,
                "status": "clean_pass"
            })
        else:
            item_copy = dict(item)
            item_copy["_validation_errors"] = errors
            corrupted.append(item_copy)
            logs.append(f"[{cid}] Initial rejection: {', '.join(errors)}")

    return {
        "clean_records": clean,
        "corrupted_records": corrupted,
        "rejection_log": logs
    }

def semantic_repair_llm(state: PipelineState) -> Dict:
    corrupted = state["corrupted_records"]
    if not corrupted:
        return {"repaired_records": []}

    prompt = [
        SystemMessage(content=(
            "You are an automated ETL repair engine. Inspect the invalid records and extract missing values "
            "from the 'notes' field. You must output ONLY a valid JSON array of objects with keys: "
            "'customer_id', 'age' (integer), 'monthly_charges' (float), 'contract_type' (must be exactly "
            "'month-to-month', 'one-year', or 'two-year'). Do not include explanation text."
        )),
        HumanMessage(content=f"Corrupt records:\n{json.dumps(corrupted, indent=2)}")
    ]

    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        parsed_repaired = json.loads(content.strip())
        if isinstance(parsed_repaired, dict):
            parsed_repaired = [parsed_repaired]
    except json.JSONDecodeError:
        parsed_repaired = []

    return {"repaired_records": parsed_repaired}

def recheck_repaired_records(state: PipelineState) -> Dict:
    accepted = []
    logs = []

    for item in state.get("repaired_records", []):
        try:
            record = CustomerRecord(**item)
            item_dict = record.model_dump()
            item_dict["status"] = "repaired_pass"
            accepted.append(item_dict)
            logs.append(f"[{record.customer_id}] Pydantic pass: repair verified.")
        except ValidationError as e:
            logs.append(f"[{item.get('customer_id', 'UNKNOWN')}] Pydantic fail: {e.errors()[0]['msg']}")

    return {
        "clean_records": state["clean_records"] + accepted,
        "rejection_log": logs
    }

def audit_generator_llm(state: PipelineState) -> Dict:
    prompt = [
        SystemMessage(content="You are a Principal MLOps Engineer. Write a concise executive audit summary."),
        HumanMessage(content=(
            f"Input Records: {len(state['raw_batch'])}\n"
            f"Total Validated: {len(state['clean_records'])}\n"
            f"Rejection/Repair Log:\n" + "\n".join(state["rejection_log"])
        ))
    ]

    response = llm.invoke(prompt)
    return {"audit_report": response.content}

# ---------------------------------------------------------------------------
# 4. Assembly & Routing
# ---------------------------------------------------------------------------
def branch_on_corruption(state: PipelineState) -> Literal["semantic_repair_llm", "audit_generator_llm"]:
    return "semantic_repair_llm" if state["corrupted_records"] else "audit_generator_llm"

workflow = StateGraph(PipelineState)
workflow.add_node("deterministic_validator", deterministic_validator)
workflow.add_node("semantic_repair_llm", semantic_repair_llm)
workflow.add_node("recheck_repaired_records", recheck_repaired_records)
workflow.add_node("audit_generator_llm", audit_generator_llm)

workflow.add_edge(START, "deterministic_validator")
workflow.add_conditional_edges(
    "deterministic_validator",
    branch_on_corruption,
    {
        "semantic_repair_llm": "semantic_repair_llm",
        "audit_generator_llm": "audit_generator_llm"
    }
)
workflow.add_edge("semantic_repair_llm", "recheck_repaired_records")
workflow.add_edge("recheck_repaired_records", "audit_generator_llm")
workflow.add_edge("audit_generator_llm", END)

app = workflow.compile()

# ---------------------------------------------------------------------------
# 5. Execution Runner (With Pre- and Post-State Display)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    thread_config = {
        "configurable": {"thread_id": "batch-run-mlx-visual-state"},
        "callbacks": [langfuse_handler]
    }

    test_batch = [
        {
            "customer_id": "C-101",
            "age": 31,
            "monthly_charges": 69.90,
            "contract_type": "month-to-month",
            "notes": "Verified subscriber"
        },
        {
            "customer_id": "C-102",
            "age": "invalid",
            "monthly_charges": None,
            "contract_type": "yearly",
            "notes": "Client confirms age is 45. Standard one-year contract at $85.50 per month."
        },
        {
            "customer_id": "C-103",
            "age": -5,
            "monthly_charges": "zero",
            "contract_type": "trial",
            "notes": "Refused to share valid contract details."
        }
    ]

    initial_state = {
        "raw_batch": test_batch,
        "clean_records": [],
        "corrupted_records": [],
        "repaired_records": [],
        "rejection_log": [],
        "audit_report": ""
    }

    console.rule("[bold cyan]LangGraph Execution Visualizer[/bold cyan]")

    # 1. Print Topology
    try:
        console.print(Panel(app.get_graph().draw_ascii(), title="[bold green]Compiled Graph DAG[/bold green]", expand=False))
    except Exception:
        pass

    # 2. Print State BEFORE Execution
    console.rule("[bold yellow]STATE BEFORE EXECUTION (Initial State)[/bold yellow]")
    console.print(JSON(json.dumps(initial_state, indent=2)))

    # 3. Stream Execution Step-by-Step
    console.rule("[bold magenta]STREAMING NODE-BY-NODE EXECUTION[/bold magenta]")
    
    # We maintain the running state snapshot to inspect post-execution
    running_state = dict(initial_state)

    for step in app.stream(initial_state, config=thread_config):
        for node_name, update in step.items():
            # Update running state snapshot
            for k, v in update.items():
                if k == "rejection_log":
                    running_state[k] = running_state[k] + v
                else:
                    running_state[k] = v

            # Visual panel per node
            if node_name == "deterministic_validator":
                table = Table(title=f"Node: {node_name} (Deterministic Check)", border_style="blue")
                table.add_column("Bucket", style="bold")
                table.add_column("Count", justify="center")
                table.add_column("Customer IDs")
                
                table.add_row(
                    "[green]Clean Records[/green]", 
                    str(len(update.get("clean_records", []))),
                    ", ".join(r["customer_id"] for r in update.get("clean_records", [])) or "None"
                )
                table.add_row(
                    "[red]Corrupt Records[/red]", 
                    str(len(update.get("corrupted_records", []))),
                    ", ".join(r["customer_id"] for r in update.get("corrupted_records", [])) or "None"
                )
                console.print(table)

            elif node_name == "semantic_repair_llm":
                tree = Tree(f"[bold magenta]Node: {node_name} (Probabilistic Local MLX)[/bold magenta]")
                repaired = update.get("repaired_records", [])
                if repaired:
                    for item in repaired:
                        tree.add(f"[cyan]{item.get('customer_id')}[/cyan] -> Age: {item.get('age')} | Charges: ${item.get('monthly_charges')} | Contract: {item.get('contract_type')}")
                else:
                    tree.add("[yellow]No candidates extracted[/yellow]")
                console.print(tree)

            elif node_name == "recheck_repaired_records":
                table = Table(title=f"Node: {node_name} (Deterministic Gate)", border_style="yellow")
                table.add_column("Metric", style="bold")
                table.add_column("Value")
                table.add_row("Total Clean Pool Now", str(len(update.get("clean_records", []))))
                table.add_row("Logs Emitted", str(len(update.get("rejection_log", []))))
                console.print(table)

            elif node_name == "audit_generator_llm":
                console.print(Panel(
                    update.get("audit_report", "").strip(),
                    title=f"[bold green]Node: {node_name} (Audit Output)[/bold green]",
                    border_style="green"
                ))

    # 4. Print State AFTER Execution
    console.rule("[bold green]STATE AFTER EXECUTION (Final Graph State)[/bold green]")
    console.print(JSON(json.dumps(running_state, indent=2)))

    # Flush Langfuse metrics
    langfuse_client.flush()
    console.rule("[bold cyan]Pipeline Complete - Langfuse Traces Synced[/bold cyan]")