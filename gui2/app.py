"""AgentZ Pipeline Manager GUI (gui2).

Provides a bright, three-column interface for selecting pipelines,
configuring inputs, managing submitted runs, and visualizing execution.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import json
from flask import Flask, jsonify, render_template, request, Response
from rich.console import Console

from streaming_printer import StreamingPrinter
from pipelines.simple import SimplePipeline
from pipelines.data_scientist import DataScientistPipeline
from pipelines.web_searcher import WebSearcherPipeline
from pipelines.vanilla_chat import VanillaChatPipeline

app = Flask(__name__, template_folder="templates", static_folder="static")


@dataclass
class PipelineField:
    """Metadata describing a configurable pipeline input field."""

    id: str
    label: str
    field_type: str
    required: bool = True
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    default: Optional[str] = None
    config_key: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.field_type,
            "required": self.required,
            "placeholder": self.placeholder,
            "help_text": self.help_text,
            "default": self.default,
        }


AVAILABLE_PIPELINES: Dict[str, Dict[str, Any]] = {
    "vanilla_chat": {
        "name": "Vanilla Chat",
        "description": "Single-turn conversational agent using the vanilla chat profile.",
        "class": VanillaChatPipeline,
        "config": "pipelines/configs/vanilla_chat.yaml",
        "primary_input": "prompt",
        "fields": [
            PipelineField(
                id="prompt",
                label="Prompt",
                field_type="textarea",
                placeholder="Ask a question to the vanilla chat agent...",
                default="Hello! How are you?",
                config_key="data.prompt",
            ),
        ],
    },
    "simple": {
        "name": "Simple Web Search",
        "description": "Routes the request through a web search agent for quick answers.",
        "class": SimplePipeline,
        "config": "pipelines/configs/simple.yaml",
        "primary_input": "prompt",
        "fields": [
            PipelineField(
                id="prompt",
                label="Search or Task Prompt",
                field_type="textarea",
                placeholder="Describe the task or information you want to find...",
                default="Find the outstanding papers of ACL 2025 and summarize key details.",
                config_key="data.prompt",
            ),
        ],
    },
    "data_scientist": {
        "name": "Data Scientist",
        "description": "Iterative data science workflow with multiple specialized agents.",
        "class": DataScientistPipeline,
        "config": "pipelines/configs/data_science.yaml",
        "primary_input": "prompt",
        "fields": [
            PipelineField(
                id="prompt",
                label="Analysis Objective",
                field_type="textarea",
                placeholder="Describe the analysis you want to perform...",
                default="Analyze banana_quality.csv and report the key quality indicators.",
                config_key="data.prompt",
            ),
            PipelineField(
                id="data_path",
                label="Dataset Path",
                field_type="text",
                placeholder="Path to the dataset file (e.g. data/banana_quality.csv)",
                default="data/banana_quality.csv",
                config_key="data.path",
            ),
        ],
    },
    "web_searcher": {
        "name": "Web Researcher",
        "description": "Research workflow that combines search, synthesis, and reporting.",
        "class": WebSearcherPipeline,
        "config": "pipelines/configs/web_searcher.yaml",
        "primary_input": "prompt",
        "fields": [
            PipelineField(
                id="prompt",
                label="Research Prompt",
                field_type="textarea",
                placeholder="What topic would you like the research agent to investigate?",
                default="Find the outstanding papers of ACL 2025 and summarize their contributions.",
                config_key="data.prompt",
            ),
        ],
    },
}


def generate_run_id() -> str:
    """Create a short unique identifier for pipeline runs."""
    return uuid.uuid4().hex[:12]


def apply_config_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Apply a nested value using dotted config keys (e.g., data.prompt)."""
    if not path:
        return

    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


pipeline_runs: Dict[str, Dict[str, Any]] = {}


def extract_result_text(result: Any) -> str:
    """Best-effort conversion of pipeline result objects to text."""
    if result is None:
        return ""
    if hasattr(result, "response"):
        return str(result.response)
    if hasattr(result, "output"):
        return str(result.output)
    return str(result)


def run_pipeline_thread(run_id: str) -> None:
    """Execute a pipeline run in a background thread."""
    run_entry = pipeline_runs.get(run_id)
    if not run_entry:
        return

    pipeline_def = AVAILABLE_PIPELINES[run_entry["pipeline_id"]]
    pipeline_class = pipeline_def["class"]
    printer: StreamingPrinter = run_entry["printer"]
    stop_flag: threading.Event = run_entry["stop_flag"]

    if stop_flag.is_set():
        run_entry["status"] = "cancelled"
        printer._emit_update("cancelled", {"message": "Run cancelled before start"})
        printer.stop_streaming()
        return

    try:
        config_payload = {"config_path": pipeline_def["config"]}

        # Apply dynamic field values
        for field in pipeline_def.get("fields", []):
            if not field.config_key:
                continue
            value = run_entry["inputs"].get(field.id)
            if value in ("", None):
                value = field.default
            if value is not None:
                apply_config_value(config_payload, field.config_key, value)

        pipeline_instance = pipeline_class(config_payload)
        pipeline_instance._printer = printer

        run_entry["pipeline_instance"] = pipeline_instance
        run_entry["status"] = "running"
        run_entry["started_at"] = time.time()

        # Determine primary input fed to run_sync
        primary_key = pipeline_def.get("primary_input")
        query_input = None
        if primary_key:
            query_input = run_entry["inputs"].get(primary_key)
        if query_input in ("", None):
            # Fall back to first non-empty input, then None
            for value in run_entry["inputs"].values():
                if value not in ("", None):
                    query_input = value
                    break

        result = pipeline_instance.run_sync(query_input)

        if stop_flag.is_set():
            run_entry["status"] = "cancelled"
            printer._emit_update("cancelled", {"message": "Run cancelled"})
            run_entry["completed_at"] = time.time()
            return

        run_entry["result"] = extract_result_text(result)
        run_entry["status"] = "completed"
        run_entry["completed_at"] = time.time()
    except Exception as exc:
        run_entry["status"] = "error"
        run_entry["error"] = str(exc)
        run_entry["completed_at"] = time.time()
        printer._emit_update("error", {"message": str(exc)})
    finally:
        printer.stop_streaming()


@app.route("/")
def index() -> str:
    """Render the main pipeline manager interface."""
    return render_template("index.html")


@app.route("/api/pipelines", methods=["GET"])
def list_pipelines():
    """Return available pipelines and their input schemas."""
    payload = []
    for pipeline_id, details in AVAILABLE_PIPELINES.items():
        fields = [field.as_dict() for field in details.get("fields", [])]
        payload.append(
            {
                "id": pipeline_id,
                "name": details["name"],
                "description": details["description"],
                "fields": fields,
            }
        )
    return jsonify({"pipelines": payload})


@app.route("/api/runs", methods=["GET"])
def list_runs():
    """Return a summary of submitted runs."""
    runs = []
    for run_id, data in sorted(
        pipeline_runs.items(), key=lambda item: item[1]["created_at"], reverse=True
    ):
        runs.append(
            {
                "id": run_id,
                "pipeline_id": data["pipeline_id"],
                "pipeline_name": data["pipeline_name"],
                "status": data["status"],
                "created_at": data["created_at"],
                "started_at": data.get("started_at"),
                "completed_at": data.get("completed_at"),
                "error": data.get("error"),
            }
        )
    return jsonify({"runs": runs})


@app.route("/api/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    """Return detailed information for a specific run."""
    run_entry = pipeline_runs.get(run_id)
    if not run_entry:
        return jsonify({"error": "Run not found"}), 404

    return jsonify(
        {
            "id": run_id,
            "pipeline_id": run_entry["pipeline_id"],
            "pipeline_name": run_entry["pipeline_name"],
            "status": run_entry["status"],
            "inputs": run_entry["inputs"],
            "result": run_entry.get("result"),
            "error": run_entry.get("error"),
            "created_at": run_entry["created_at"],
            "started_at": run_entry.get("started_at"),
            "completed_at": run_entry.get("completed_at"),
        }
    )


@app.route("/api/runs", methods=["POST"])
def create_run():
    """Create a new pipeline run with provided inputs."""
    payload = request.json or {}
    pipeline_id = payload.get("pipeline_id")
    inputs = payload.get("inputs", {})

    if pipeline_id not in AVAILABLE_PIPELINES:
        return jsonify({"error": "Invalid pipeline ID"}), 400

    pipeline_def = AVAILABLE_PIPELINES[pipeline_id]
    errors: List[str] = []

    for field in pipeline_def.get("fields", []):
        if field.required and not inputs.get(field.id):
            errors.append(f"{field.label} is required")

    if errors:
        return jsonify({"error": "Validation error", "details": errors}), 400

    run_id = generate_run_id()
    console = Console()
    printer = StreamingPrinter(console)
    stop_flag = threading.Event()

    run_entry = {
        "id": run_id,
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_def["name"],
        "inputs": inputs,
        "status": "queued",
        "created_at": time.time(),
        "printer": printer,
        "stop_flag": stop_flag,
        "thread": None,
        "pipeline_instance": None,
        "result": None,
        "error": None,
    }

    pipeline_runs[run_id] = run_entry

    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(run_id,),
        daemon=True,
        name=f"pipeline-run-{run_id}",
    )
    run_entry["thread"] = thread
    thread.start()

    return jsonify(
        {
            "run": {
                "id": run_id,
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_def["name"],
                "status": run_entry["status"],
                "created_at": run_entry["created_at"],
            }
        }
    ), 201


@app.route("/api/runs/<run_id>/stop", methods=["POST"])
def stop_run(run_id: str):
    """Attempt to cancel an active run."""
    run_entry = pipeline_runs.get(run_id)
    if not run_entry:
        return jsonify({"error": "Run not found"}), 404

    stop_flag: threading.Event = run_entry["stop_flag"]
    stop_flag.set()

    pipeline_instance = run_entry.get("pipeline_instance")
    if pipeline_instance and hasattr(pipeline_instance, "stop"):
        try:
            pipeline_instance.stop()
        except Exception:
            pass

    run_entry["status"] = "cancelling"

    return jsonify({"status": "cancelling"})


@app.route("/api/runs/<run_id>/stream", methods=["GET"])
def stream_run(run_id: str):
    """Stream updates for a specific run via Server-Sent Events."""
    run_entry = pipeline_runs.get(run_id)
    if not run_entry:
        return jsonify({"error": "Run not found"}), 404

    printer: StreamingPrinter = run_entry["printer"]

    def event_stream():
        while True:
            if run_id not in pipeline_runs:
                break

            update = printer.get_updates(timeout=0.3)

            if update:
                payload = {
                    "event": update["type"],
                    "payload": update.get("data", {}),
                    "run_status": pipeline_runs[run_id]["status"],
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if update["type"] == "stream_end":
                    break
            else:
                status = pipeline_runs[run_id]["status"]
                if status in {"completed", "error", "cancelled"}:
                    break
                time.sleep(0.1)

        # Final snapshot
        final = pipeline_runs.get(run_id)
        if final:
            summary = {
                "event": "summary",
                "payload": {
                    "status": final["status"],
                    "result": final.get("result"),
                    "error": final.get("error"),
                },
                "run_status": final["status"],
            }
            yield f"data: {json.dumps(summary)}\n\n"

        return

    # Use text/event-stream mimetype for SSE
    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentZ GUI 2 Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=9090, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()

    print("🚀 Starting AgentZ GUI 2 Server")
    print(f"📍 http://{args.host}:{args.port}")
    print("Available pipelines:")
    for pipeline_id, details in AVAILABLE_PIPELINES.items():
        print(f"  - {details['name']} ({pipeline_id})")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
