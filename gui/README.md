# AgentZ GUI - Web Interface

A ChatGPT-style web interface for interacting with AgentZ pipelines.

## Features

- 🎨 **Modern ChatGPT-like Interface** - Clean, intuitive design with dark theme
- 🔄 **Pipeline Selection** - Switch between different AI pipelines on the fly
- 💬 **Real-time Chat** - Stream responses as they're generated
- 📝 **Chat History** - Maintains conversation context
- 🚀 **Multiple Pipelines** - Supports Web Searcher, Data Scientist, and Simple Assistant

## Quick Start

### Installation

Make sure you have Flask installed:

```bash
pip install flask
```

Or if using the project's virtual environment:

```bash
source .venv/bin/activate
pip install flask
```

### Running the Server

From the `gui` directory:

```bash
python app.py
```

Or with custom settings:

```bash
# Custom port
python app.py --port 8080

# Different host
python app.py --host 0.0.0.0 --port 9999

# Debug mode (auto-reload on changes)
python app.py --debug
```

### Default Access

Once started, open your browser and navigate to:

```
http://localhost:9999
```

## Usage

1. **Select a Pipeline**
   - Use the dropdown in the sidebar to choose a pipeline
   - Options: Web Searcher, Data Scientist, or Simple Assistant
   - Each pipeline has different capabilities

2. **Start Chatting**
   - Type your message in the input box at the bottom
   - Press Enter to send (Shift+Enter for new line)
   - Watch as the AI responds in real-time

3. **Manage Conversations**
   - Click "Clear" to reset the current chat
   - Click "+ New Chat" to start fresh
   - Switch pipelines to use different AI capabilities

## Available Pipelines

### Web Searcher
Search the web for information and research topics. Great for finding recent information and extracting data.

**Example prompts:**
- "Find the outstanding papers of ACL 2025"
- "Search for latest developments in quantum computing"
- "What are the top AI conferences in 2025?"

### Data Scientist
Analyze datasets and build machine learning models. Upload CSV files and get insights.

**Example prompts:**
- "Analyze the banana quality dataset"
- "Build a classification model for this data"
- "Show me the correlation between features"

### Simple Assistant
General purpose AI assistant for various tasks.

**Example prompts:**
- "Explain quantum computing in simple terms"
- "Write a Python function to sort a list"
- "Help me debug this code"

## Project Structure

```
gui/
├── app.py                 # Flask application server
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Styling (ChatGPT-inspired)
│   └── js/
│       └── chat.js       # Frontend JavaScript
└── README.md             # This file
```

## API Endpoints

The GUI provides several REST API endpoints:

- `GET /` - Main interface
- `GET /api/pipelines` - List available pipelines
- `POST /api/select-pipeline` - Select a pipeline
- `POST /api/chat` - Send message and stream response
- `GET /api/history` - Get chat history
- `POST /api/clear` - Clear chat history

## Configuration

### Adding New Pipelines

Edit `app.py` and add to the `AVAILABLE_PIPELINES` dictionary:

```python
AVAILABLE_PIPELINES = {
    "your_pipeline": {
        "name": "Your Pipeline Name",
        "description": "What your pipeline does",
        "class": YourPipelineClass,
        "config": {
            "config_path": "path/to/config.yaml"
        }
    }
}
```

### Customizing the UI

- **Colors/Theme**: Edit `static/css/style.css`
- **Layout**: Modify `templates/index.html`
- **Behavior**: Update `static/js/chat.js`

## Troubleshooting

### Port Already in Use

If port 9999 is already in use, specify a different port:

```bash
python app.py --port 8888
```

### Pipeline Not Working

1. Check that the pipeline configuration file exists
2. Verify environment variables (e.g., `GEMINI_API_KEY`)
3. Look at the server console for error messages

### Connection Issues

If you can't connect to the server:

1. Check that the server is running
2. Verify the correct host:port combination
3. Check firewall settings if accessing remotely

## Development

### Debug Mode

Enable debug mode for development:

```bash
python app.py --debug
```

This enables:
- Auto-reload on file changes
- Detailed error messages
- Flask debug toolbar

### Adding Features

The code is structured for easy extension:

- **Backend**: Add routes in `app.py`
- **Frontend**: Add UI in `templates/index.html`
- **Styling**: Add CSS in `static/css/style.css`
- **Logic**: Add JavaScript in `static/js/chat.js`

## Requirements

- Python 3.11+
- Flask
- AgentZ pipelines properly configured
- Environment variables set (API keys, etc.)

## License

Same as the parent AgentZ project.
