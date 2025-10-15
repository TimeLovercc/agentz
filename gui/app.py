"""GUI Application for AgentZ - ChatGPT-style Interface with Pipeline Selection

Fully integrated with the agentz framework for real pipeline execution.
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from typing import Dict, Any, Optional
from rich.console import Console

from gui.streaming_printer import StreamingPrinter
from pipelines.simple import SimplePipeline
from pipelines.data_scientist import DataScientistPipeline
from pipelines.web_searcher import WebSearcherPipeline
from pipelines.vanilla_chat import VanillaChatPipeline
from pipelines.vanilla_chat_persistent import VanillaChatPersistentPipeline

app = Flask(__name__)

# Store chat history and active pipeline sessions
chat_sessions: Dict[str, Any] = {
    "messages": [],
    "current_pipeline": None,
    "current_session_id": None,
}

# Active pipeline sessions: session_id -> {pipeline, printer, thread, result, stop_flag}
pipeline_sessions: Dict[str, Dict[str, Any]] = {}

# Stop flags for cancellation: session_id -> threading.Event
stop_flags: Dict[str, threading.Event] = {}


# Available pipelines configuration
AVAILABLE_PIPELINES = {
    "vanilla_chat": {
        "name": "Vanilla Chat (Persistent)",
        "description": "Multi-turn conversational agent",
        "class": VanillaChatPersistentPipeline,
        "config": "pipelines/configs/vanilla_chat.yaml",
        "persistent": True,  # Mark as persistent
    },
    "simple": {
        "name": "Simple Web Search",
        "description": "Single-pass web search pipeline",
        "class": SimplePipeline,
        "config": "pipelines/configs/simple.yaml",
    },
    "data_scientist": {
        "name": "Data Scientist",
        "description": "Advanced data analysis with iterative refinement",
        "class": DataScientistPipeline,
        "config": "pipelines/configs/data_science.yaml",
    },
    "web_searcher": {
        "name": "Web Searcher",
        "description": "Research assistant with web search capabilities",
        "class": WebSearcherPipeline,
        "config": "pipelines/configs/web_searcher.yaml",
    },
}


def run_pipeline_in_thread(
    pipeline_class,
    config: Dict[str, Any],
    user_message: str,
    session_id: str,
    printer: StreamingPrinter,
    stop_flag: threading.Event
):
    """Run pipeline in a separate thread with custom printer.

    Args:
        pipeline_class: Pipeline class to instantiate
        config: Configuration dictionary for the pipeline
        user_message: User's query/message
        session_id: Unique session identifier
        printer: StreamingPrinter instance for capturing updates
        stop_flag: Threading event for cancellation
    """
    try:
        # Check if stopped before starting
        if stop_flag.is_set():
            pipeline_sessions[session_id]["status"] = "cancelled"
            printer._emit_update("cancelled", {"message": "Pipeline execution cancelled"})
            printer.stop_streaming()
            return

        # Create pipeline with custom printer
        pipeline = pipeline_class(config)

        # Replace the pipeline's printer with our streaming printer
        pipeline._printer = printer

        # Store pipeline reference for potential cleanup
        pipeline_sessions[session_id]["pipeline"] = pipeline

        # Check stop flag periodically during execution
        # Note: This is a simple approach. For better cancellation,
        # we'd need to integrate stop checking into the pipeline itself

        # Run the pipeline synchronously
        result = pipeline.run_sync(user_message)

        print(f"[DEBUG] Pipeline result type: {type(result)}")
        print(f"[DEBUG] Pipeline result: {result}")

        # Check if stopped after execution
        if stop_flag.is_set():
            pipeline_sessions[session_id]["status"] = "cancelled"
            printer._emit_update("cancelled", {"message": "Pipeline execution cancelled"})
            printer.stop_streaming()
            return

        # Store result
        pipeline_sessions[session_id]["result"] = result
        pipeline_sessions[session_id]["status"] = "completed"

        print(f"[DEBUG] Result stored in session: {pipeline_sessions[session_id].get('result')}")

        # Signal completion
        printer.stop_streaming()

    except Exception as e:
        # Check if it's due to cancellation
        if stop_flag.is_set():
            pipeline_sessions[session_id]["status"] = "cancelled"
            printer._emit_update("cancelled", {"message": "Pipeline execution cancelled"})
        else:
            # Store error
            error_msg = f"Pipeline error: {str(e)}"
            pipeline_sessions[session_id]["error"] = error_msg
            pipeline_sessions[session_id]["status"] = "error"

            # Emit error event
            printer._emit_update("error", {"message": error_msg})

        printer.stop_streaming()


@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html', pipelines=AVAILABLE_PIPELINES)


@app.route('/api/pipelines', methods=['GET'])
def get_pipelines():
    """Get list of available pipelines"""
    pipelines = []
    for key, config in AVAILABLE_PIPELINES.items():
        pipelines.append({
            "id": key,
            "name": config["name"],
            "description": config["description"]
        })
    return jsonify({"pipelines": pipelines})


@app.route('/api/select-pipeline', methods=['POST'])
def select_pipeline():
    """Select and initialize a pipeline"""
    data = request.json
    pipeline_id = data.get('pipeline_id')

    if pipeline_id not in AVAILABLE_PIPELINES:
        return jsonify({"error": "Invalid pipeline ID"}), 400

    # Clean up previous persistent pipeline if any
    prev_pipeline_id = chat_sessions.get("current_pipeline")
    if prev_pipeline_id:
        prev_session_id = f"{prev_pipeline_id}_persistent"
        if prev_session_id in pipeline_sessions:
            print(f"[DEBUG] Cleaning up previous persistent pipeline: {prev_session_id}")
            session = pipeline_sessions[prev_session_id]
            pipeline_instance = session.get("pipeline")
            if pipeline_instance and hasattr(pipeline_instance, 'stop'):
                pipeline_instance.stop()
            pipeline_sessions.pop(prev_session_id, None)
            stop_flags.pop(prev_session_id, None)

    # Store selected pipeline
    chat_sessions["current_pipeline"] = pipeline_id

    # Clear previous messages when switching pipelines
    chat_sessions["messages"] = []

    return jsonify({
        "status": "success",
        "pipeline": AVAILABLE_PIPELINES[pipeline_id]["name"]
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and stream responses"""
    data = request.json
    user_message = data.get('message', '')
    pipeline_id = chat_sessions.get("current_pipeline")

    if not pipeline_id:
        return jsonify({"error": "No pipeline selected"}), 400

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    # Add user message to history
    chat_sessions["messages"].append({
        "role": "user",
        "content": user_message
    })

    # Get pipeline configuration
    pipeline_config_dict = AVAILABLE_PIPELINES[pipeline_id]
    is_persistent = pipeline_config_dict.get("persistent", False)

    # Check if we have an existing persistent pipeline session
    persistent_session_id = f"{pipeline_id}_persistent"
    existing_session = pipeline_sessions.get(persistent_session_id) if is_persistent else None

    if is_persistent and existing_session and existing_session.get("status") == "running":
        # Reuse existing persistent pipeline
        print(f"[DEBUG] Reusing persistent pipeline: {persistent_session_id}")
        session_id = persistent_session_id
        pipeline_instance = existing_session["pipeline"]
        printer = existing_session["printer"]

        # Add message to pipeline's input queue
        pipeline_instance.add_user_message(user_message)

    else:
        # Create new pipeline session
        if is_persistent:
            session_id = persistent_session_id
            print(f"[DEBUG] Creating new persistent pipeline: {session_id}")
        else:
            session_id = f"{pipeline_id}_{int(time.time() * 1000)}"
            print(f"[DEBUG] Creating one-shot pipeline: {session_id}")

        chat_sessions["current_session_id"] = session_id

        pipeline_class = pipeline_config_dict["class"]
        config_path = pipeline_config_dict["config"]
        config = {
            "config_path": config_path,
            "prompt": user_message,
        }

        # Create streaming printer with a console
        console = Console()
        printer = StreamingPrinter(console)

        # Create stop flag for this session
        stop_flag = threading.Event()
        stop_flags[session_id] = stop_flag

        # Initialize session
        pipeline_sessions[session_id] = {
            "pipeline_id": pipeline_id,
            "printer": printer,
            "status": "running",
            "result": None,
            "error": None,
            "pipeline": None,
            "stop_flag": stop_flag,
        }

        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_pipeline_in_thread,
            args=(pipeline_class, config, user_message, session_id, printer, stop_flag),
            daemon=True
        )
        pipeline_sessions[session_id]["thread"] = thread
        thread.start()

    # Stream updates via SSE
    def generate_response():
        """Generator function for streaming response"""
        try:
            assistant_response = ""

            while True:
                # Get updates from printer queue
                update = printer.get_updates(timeout=0.1)

                if update:
                    event_type = update.get("type")
                    event_data = update.get("data", {})
                    print(f"[DEBUG] SSE received event: {event_type}")

                    if event_type == "group_end":
                        # For persistent pipelines, each iteration's end means a response is ready
                        # Check if we have a result to send
                        session = pipeline_sessions.get(session_id, {})
                        is_persistent = AVAILABLE_PIPELINES.get(
                            session.get("pipeline_id"), {}
                        ).get("persistent", False)

                        if is_persistent:
                            # For persistent pipeline, check if there's a completed response
                            pipeline_instance = session.get("pipeline")
                            if pipeline_instance and hasattr(pipeline_instance, 'state'):
                                # Get the latest response from state
                                response_text = getattr(pipeline_instance.state, 'final_report', None)
                                if response_text and response_text != assistant_response:
                                    print(f"[DEBUG] Group end - new response available: {response_text}")
                                    assistant_response = response_text

                                    # Add to chat history
                                    chat_sessions["messages"].append({
                                        "role": "assistant",
                                        "content": assistant_response
                                    })

                                    # Send response to frontend
                                    yield f"data: {json.dumps({'content': assistant_response, 'done': True})}\n\n"

                                    # For persistent pipeline, keep session alive but mark this turn complete
                                    print(f"[DEBUG] Turn complete, pipeline stays alive")
                                    break  # Exit SSE for this request, but keep pipeline running

                    elif event_type == "stream_end":
                        # Pipeline finished completely (non-persistent or stopped)
                        session = pipeline_sessions.get(session_id, {})

                        print(f"[DEBUG] Stream end received, session: {session.keys() if session else 'None'}")

                        is_persistent = AVAILABLE_PIPELINES.get(
                            session.get("pipeline_id"), {}
                        ).get("persistent", False)

                        if session.get("error"):
                            error_msg = session["error"]
                            print(f"[DEBUG] Error in session: {error_msg}")
                            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
                        else:
                            result = session.get("result")
                            print(f"[DEBUG] Result from session: {result}")
                            print(f"[DEBUG] Result type: {type(result)}")

                            if result:
                                # Extract text from result
                                # Handle different output formats (ChatOutput has .response, ToolAgentOutput has .output)
                                if hasattr(result, 'response'):
                                    response_text = str(result.response)
                                    print(f"[DEBUG] Extracted response: {response_text}")
                                elif hasattr(result, 'output'):
                                    response_text = str(result.output)
                                    print(f"[DEBUG] Extracted output: {response_text}")
                                else:
                                    response_text = str(result)
                                    print(f"[DEBUG] Converted to string: {response_text}")

                                # Only send if we haven't already sent it
                                if response_text != assistant_response:
                                    assistant_response = response_text

                                    # Add to chat history
                                    chat_sessions["messages"].append({
                                        "role": "assistant",
                                        "content": assistant_response
                                    })

                                    print(f"[DEBUG] Yielding final response: {assistant_response}")
                                    yield f"data: {json.dumps({'content': assistant_response, 'done': True})}\n\n"
                            else:
                                print(f"[DEBUG] No result found, yielding default message")
                                yield f"data: {json.dumps({'content': 'Pipeline completed', 'done': True})}\n\n"

                        # Clean up session only if not persistent
                        if not is_persistent:
                            print(f"[DEBUG] Cleaning up non-persistent session")
                            pipeline_sessions.pop(session_id, None)
                            stop_flags.pop(session_id, None)
                        else:
                            print(f"[DEBUG] Keeping persistent session alive")

                        break

                    elif event_type == "error":
                        error_msg = event_data.get("message", "Unknown error")
                        yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
                        pipeline_sessions.pop(session_id, None)
                        stop_flags.pop(session_id, None)
                        break

                    elif event_type == "cancelled":
                        # Pipeline was cancelled
                        cancel_msg = event_data.get("message", "Pipeline cancelled")
                        yield f"data: {json.dumps({'cancelled': cancel_msg, 'done': True})}\n\n"
                        pipeline_sessions.pop(session_id, None)
                        stop_flags.pop(session_id, None)
                        break

                    else:
                        # Stream progress update
                        yield f"data: {json.dumps({'update': event_data, 'done': False})}\n\n"
                else:
                    # No update received
                    print(f"[DEBUG] No update from queue, session exists: {session_id in pipeline_sessions}")

                # Check if session still exists
                if session_id not in pipeline_sessions:
                    print(f"[DEBUG] Session no longer exists, breaking loop")
                    break

                # Small delay to prevent busy loop
                time.sleep(0.05)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
            pipeline_sessions.pop(session_id, None)
            stop_flags.pop(session_id, None)

    return Response(generate_response(), mimetype='text/event-stream')


@app.route('/api/stop', methods=['POST'])
def stop_pipeline():
    """Stop the currently running pipeline"""
    session_id = chat_sessions.get("current_session_id")

    if not session_id:
        return jsonify({"error": "No active pipeline session"}), 400

    if session_id not in pipeline_sessions:
        return jsonify({"error": "Pipeline session not found"}), 404

    # Set the stop flag
    stop_flag = stop_flags.get(session_id)
    if stop_flag:
        stop_flag.set()

        # Update session status
        if session_id in pipeline_sessions:
            pipeline_sessions[session_id]["status"] = "stopping"

        return jsonify({
            "status": "success",
            "message": "Stop signal sent to pipeline"
        })
    else:
        return jsonify({"error": "Stop flag not found"}), 404


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    return jsonify({"messages": chat_sessions["messages"]})


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    chat_sessions["messages"] = []
    return jsonify({"status": "success"})


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AgentZ GUI Server')
    parser.add_argument('--port', type=int, default=9999, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    print(f"\n🚀 Starting AgentZ GUI Server...")
    print(f"📍 Server running at: http://{args.host}:{args.port}")
    print(f"🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"🤖 Integrated with AgentZ Framework\n")
    print(f"Available Pipelines:")
    for key, config in AVAILABLE_PIPELINES.items():
        print(f"  - {config['name']}: {config['description']}")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
