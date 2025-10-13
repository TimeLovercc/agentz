# Quick Start Guide - AgentZ GUI

## Running the Server

### Option 1: Using the start script
```bash
cd gui
./start.sh
```

### Option 2: Direct Python command
```bash
cd gui
source ../.venv/bin/activate
python app.py --port 9999
```

### Option 3: Custom port
```bash
cd gui
source ../.venv/bin/activate
python app.py --port 8080
```

## Accessing the Interface

Once the server is running, open your browser and go to:
```
http://localhost:9999
```

## Using the GUI

1. **Select a Pipeline**: Choose from the dropdown (Pipeline Alpha, Beta, or Gamma)
2. **Start Chatting**: Type your message in the input box
3. **Send Message**: Press Enter or click the send button
4. **View Response**: The bot will always reply with "Hi" (demo mode)

## Features

- ✅ ChatGPT-style interface with dark theme
- ✅ Three demo pipelines in dropdown
- ✅ Real-time message streaming
- ✅ Chat history
- ✅ Clear chat functionality
- ✅ Responsive design

## Demo Mode Note

This is a simplified demo version:
- The chatbot always replies "Hi" regardless of input
- Pipeline selection doesn't change the behavior (all reply "Hi")
- No actual AI backend is connected

This provides a working UI/UX prototype that can be connected to real backends later.

## Troubleshooting

### Flask not installed
```bash
cd .. # Go to project root
source .venv/bin/activate
uv add flask
```

### Port already in use
Use a different port:
```bash
python app.py --port 8888
```

### Can't connect
Check that:
1. The server is running (you should see startup messages)
2. You're using the correct URL (http://localhost:9999)
3. No firewall is blocking the connection
