"""GUI Application for AgentZ - ChatGPT-style Interface with Pipeline Selection
DEMO VERSION: Simulated chatbot that always replies "Hi"
"""

import json
import time
from flask import Flask, render_template, request, jsonify, Response
from typing import Dict, Any

app = Flask(__name__)

# Store chat history and active pipeline
chat_sessions: Dict[str, Any] = {
    "messages": [],
    "current_pipeline": None
}

# Available pipelines configuration (demo pipelines)
AVAILABLE_PIPELINES = {
    "pipeline_alpha": {
        "name": "Pipeline Alpha",
        "description": "First demo pipeline for testing"
    },
    "pipeline_beta": {
        "name": "Pipeline Beta",
        "description": "Second demo pipeline for testing"
    },
    "pipeline_gamma": {
        "name": "Pipeline Gamma",
        "description": "Third demo pipeline for testing"
    }
}


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

    def generate_response():
        """Generator function for streaming response"""
        try:
            # Simulate processing delay
            time.sleep(0.5)

            # Simple demo response - always reply "Hi"
            response_text = "Hi"

            # Add assistant response to history
            chat_sessions["messages"].append({
                "role": "assistant",
                "content": response_text
            })

            # Stream the response
            yield f"data: {json.dumps({'content': response_text, 'done': False})}\n\n"
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

    return Response(generate_response(), mimetype='text/event-stream')


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

    parser = argparse.ArgumentParser(description='AgentZ GUI Server (Demo)')
    parser.add_argument('--port', type=int, default=9999, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    print(f"\n🚀 Starting AgentZ GUI Server (Demo Mode)...")
    print(f"📍 Server running at: http://{args.host}:{args.port}")
    print(f"🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"💬 Demo: Chatbot always replies 'Hi'\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
