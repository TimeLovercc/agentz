#!/bin/bash
# Quick start script for AgentZ GUI

cd "$(dirname "$0")"

# Activate virtual environment
source ../.venv/bin/activate

# Start the server
python app.py --port 9999

# Default: http://localhost:9999
