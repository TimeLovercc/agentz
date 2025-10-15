// Chat Application JavaScript

// State management
const state = {
    currentPipeline: null,
    isProcessing: false,
    messages: []
};

// DOM Elements
const elements = {
    pipelineSelect: document.getElementById('pipeline-select'),
    pipelineDescription: document.getElementById('pipeline-description'),
    currentPipelineName: document.getElementById('current-pipeline-name'),
    messagesContainer: document.getElementById('messages-container'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    stopBtn: document.getElementById('stop-btn'),
    clearBtn: document.getElementById('clear-chat-btn'),
    newChatBtn: document.getElementById('new-chat-btn'),
    statusText: document.getElementById('status-text')
};

// Pipeline descriptions
const pipelineDescriptions = {
    'vanilla_chat': 'Multi-turn conversational agent (persistent session)',
    'web_searcher': 'Search the web for information and research topics',
    'data_scientist': 'Analyze datasets and build machine learning models',
    'simple': 'General purpose AI assistant for various tasks'
};

// Initialize the application
function init() {
    setupEventListeners();
    loadChatHistory();
}

// Setup event listeners
function setupEventListeners() {
    // Pipeline selection
    elements.pipelineSelect.addEventListener('change', handlePipelineChange);

    // Send message
    elements.sendBtn.addEventListener('click', handleSendMessage);

    // Stop pipeline execution
    elements.stopBtn.addEventListener('click', handleStopPipeline);

    // Enter key to send (Shift+Enter for new line)
    elements.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-resize textarea
    elements.userInput.addEventListener('input', autoResizeTextarea);

    // Clear chat
    elements.clearBtn.addEventListener('click', handleClearChat);

    // New chat
    elements.newChatBtn.addEventListener('click', handleNewChat);
}

// Handle pipeline selection change
async function handlePipelineChange() {
    const pipelineId = elements.pipelineSelect.value;

    if (!pipelineId) {
        disableChatInput();
        elements.pipelineDescription.textContent = '';
        elements.currentPipelineName.textContent = 'AgentZ Assistant';
        return;
    }

    // Update UI
    const pipelineName = elements.pipelineSelect.options[elements.pipelineSelect.selectedIndex].text;
    elements.currentPipelineName.textContent = pipelineName;
    elements.pipelineDescription.textContent = pipelineDescriptions[pipelineId] || '';

    // Select pipeline on backend
    try {
        const response = await fetch('/api/select-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pipeline_id: pipelineId })
        });

        const data = await response.json();

        if (response.ok) {
            state.currentPipeline = pipelineId;
            enableChatInput();
            elements.statusText.textContent = `${pipelineName} ready`;

            // Clear welcome message
            if (elements.messagesContainer.querySelector('.welcome-message')) {
                elements.messagesContainer.innerHTML = '';
            }
        } else {
            showError(data.error || 'Failed to select pipeline');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

// Handle send message
async function handleSendMessage() {
    const message = elements.userInput.value.trim();

    if (!message || state.isProcessing || !state.currentPipeline) {
        return;
    }

    // Add user message to UI
    addMessage('user', message);

    // Clear input
    elements.userInput.value = '';
    autoResizeTextarea();

    // Disable input while processing
    state.isProcessing = true;
    disableChatInput();
    showStopButton();
    elements.statusText.textContent = 'Processing...';

    // Show typing indicator
    const typingId = addTypingIndicator();

    // Send to backend
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Request failed');
        }

        // Read streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMessage = '';

        // Remove typing indicator
        removeTypingIndicator(typingId);

        // Create assistant message element
        const messageId = addMessage('assistant', '');

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.error) {
                            updateMessage(messageId, `Error: ${data.error}`);
                            break;
                        }

                        if (data.cancelled) {
                            updateMessage(messageId, `Cancelled: ${data.cancelled}`);
                            break;
                        }

                        if (data.content) {
                            assistantMessage += data.content;
                            updateMessage(messageId, assistantMessage);
                        }

                        if (data.done) {
                            break;
                        }
                    } catch (e) {
                        // Ignore JSON parse errors for incomplete chunks
                    }
                }
            }
        }

    } catch (error) {
        removeTypingIndicator(typingId);
        showError('Error: ' + error.message);
    } finally {
        state.isProcessing = false;
        hideStopButton();
        enableChatInput();
        elements.statusText.textContent = 'Ready';
    }
}

// Handle stop pipeline
async function handleStopPipeline() {
    if (!state.isProcessing) {
        return;
    }

    elements.statusText.textContent = 'Stopping...';

    try {
        const response = await fetch('/api/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (response.ok) {
            elements.statusText.textContent = 'Stopped';
        } else {
            showError(data.error || 'Failed to stop pipeline');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

// Add message to chat
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const messageId = `msg-${Date.now()}-${Math.random()}`;
    messageDiv.id = messageId;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const roleDiv = document.createElement('div');
    roleDiv.className = 'message-role';
    roleDiv.textContent = role === 'user' ? 'You' : 'Assistant';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = content;

    contentDiv.appendChild(roleDiv);
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(contentDiv);

    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom();

    // Add to state
    state.messages.push({ role, content });

    return messageId;
}

// Update message content
function updateMessage(messageId, content) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const textDiv = messageDiv.querySelector('.message-text');
        textDiv.textContent = content;
        scrollToBottom();
    }
}

// Add typing indicator
function addTypingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'message assistant';

    const indicatorId = `typing-${Date.now()}`;
    indicatorDiv.id = indicatorId;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const roleDiv = document.createElement('div');
    roleDiv.className = 'message-role';
    roleDiv.textContent = 'Assistant';

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';

    contentDiv.appendChild(roleDiv);
    contentDiv.appendChild(typingDiv);
    indicatorDiv.appendChild(contentDiv);

    elements.messagesContainer.appendChild(indicatorDiv);
    scrollToBottom();

    return indicatorId;
}

// Remove typing indicator
function removeTypingIndicator(indicatorId) {
    const indicator = document.getElementById(indicatorId);
    if (indicator) {
        indicator.remove();
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    elements.messagesContainer.appendChild(errorDiv);
    scrollToBottom();

    // Remove after 5 seconds
    setTimeout(() => errorDiv.remove(), 5000);
}

// Enable chat input
function enableChatInput() {
    elements.userInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.userInput.focus();
}

// Disable chat input
function disableChatInput() {
    elements.userInput.disabled = true;
    elements.sendBtn.disabled = true;
}

// Show stop button
function showStopButton() {
    elements.sendBtn.style.display = 'none';
    elements.stopBtn.style.display = 'flex';
}

// Hide stop button
function hideStopButton() {
    elements.stopBtn.style.display = 'none';
    elements.sendBtn.style.display = 'flex';
}

// Auto-resize textarea
function autoResizeTextarea() {
    elements.userInput.style.height = 'auto';
    elements.userInput.style.height = elements.userInput.scrollHeight + 'px';
}

// Scroll to bottom of messages
function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// Handle clear chat
async function handleClearChat() {
    if (!confirm('Clear all messages?')) {
        return;
    }

    try {
        await fetch('/api/clear', { method: 'POST' });
        elements.messagesContainer.innerHTML = '';
        state.messages = [];
        elements.statusText.textContent = 'Chat cleared';
    } catch (error) {
        showError('Failed to clear chat: ' + error.message);
    }
}

// Handle new chat
function handleNewChat() {
    handleClearChat();
}

// Load chat history
async function loadChatHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        if (data.messages && data.messages.length > 0) {
            elements.messagesContainer.innerHTML = '';
            data.messages.forEach(msg => {
                addMessage(msg.role, msg.content);
            });
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
