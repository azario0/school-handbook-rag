from flask import Flask, render_template, request, jsonify, session
import os
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import re
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Configuration
GEMINI_API_KEY = 'YOUR_API_KEY'  # Move to environment variable in production
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.0-flash-exp"  # For generating responses
FAISS_INDEX_PATH = "school_handbook.faiss"
CHUNKS_DATA_PATH = "school_handbook_chunks.pkl"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Global variables for loaded data
loaded_index = None
loaded_chunks = []

def load_rag_data():
    """Load FAISS index and chunks data"""
    global loaded_index, loaded_chunks
    try:
        loaded_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_DATA_PATH, "rb") as f:
            loaded_chunks = pickle.load(f)
        print(f"✅ Loaded FAISS index with {loaded_index.ntotal} vectors and {len(loaded_chunks)} chunks")
        return True
    except Exception as e:
        print(f"❌ Error loading RAG data: {e}")
        return False

def retrieve_relevant_chunks(user_prompt: str, top_k: int = 3):
    """Retrieve the most relevant chunks from the FAISS index"""
    if not loaded_index or not loaded_chunks:
        return [], []
    
    try:
        # Embed the user prompt
        query_embedding_response = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=user_prompt,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = query_embedding_response['embedding']
        query_vector_np = np.array([query_embedding]).astype('float32')
        
        # Search the FAISS index
        distances, indices = loaded_index.search(query_vector_np, top_k)
        
        relevant_chunks = []
        chunk_info = []
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx < len(loaded_chunks):
                chunk_text = loaded_chunks[idx]
                relevant_chunks.append(chunk_text)
                chunk_info.append({
                    'index': int(idx),
                    'distance': float(dist),
                    'preview': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                })
        
        return relevant_chunks, chunk_info
    
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return [], []

def generate_response(user_question: str, relevant_chunks: list):
    """Generate a response using Gemini with the retrieved context"""
    if not relevant_chunks:
        return "I couldn't find relevant information in the school handbook to answer your question. Please try rephrasing your question or contact the school administration directly."
    
    # Prepare context from retrieved chunks
    context = "\n\n".join([f"Source {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
    
    # Create prompt for Gemini
    prompt = f"""You are a helpful assistant for Pathways Academy. Use the following information from the school handbook to answer the student's question accurately and helpfully.

CONTEXT FROM SCHOOL HANDBOOK:
{context}

STUDENT QUESTION: {user_question}

INSTRUCTIONS:
- Answer based primarily on the provided context from the school handbook
- Be clear, helpful, and student-friendly
- If the context doesn't fully answer the question, mention what information is available and suggest contacting school administration for additional details
- Use a warm, supportive tone appropriate for students and parents
- Structure your response clearly with headings or bullet points when appropriate

ANSWER:"""

    try:
        model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again later or contact the school directly."

@app.route('/')
def index():
    """Main page"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['chat_history'] = []
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question asking"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Retrieve relevant chunks
        relevant_chunks, chunk_info = retrieve_relevant_chunks(question, top_k=3)
        
        # Generate response
        response = generate_response(question, relevant_chunks)
        
        # Store in session history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        chat_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response': response,
            'sources_used': len(relevant_chunks)
        }
        
        session['chat_history'].append(chat_entry)
        session.modified = True
        
        return jsonify({
            'response': response,
            'sources_info': chunk_info,
            'timestamp': datetime.now().strftime('%I:%M %p')
        })
    
    except Exception as e:
        print(f"Error in ask_question: {e}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500

@app.route('/history')
def get_history():
    """Get chat history"""
    history = session.get('chat_history', [])
    return jsonify({'history': history})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    session['chat_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'faiss_loaded': loaded_index is not None,
        'chunks_loaded': len(loaded_chunks) > 0,
        'total_chunks': len(loaded_chunks)
    }
    return jsonify(status)

# Template for the main page
def create_templates():
    """Create the HTML template"""
    template_dir = 'templates'
    static_dir = 'static'
    
    # Create directories if they don't exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # HTML Template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pathways Academy - Student Handbook Assistant</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary-color: #f8fafc;
            --accent-color: #10b981;
            --text-color: #1f2937;
            --text-light: #6b7280;
            --border-color: #e5e7eb;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--gradient);
            min-height: 100vh;
            color: var(--text-color);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: var(--shadow);
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            max-height: 70vh;
        }

        .chat-header {
            background: var(--primary-color);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f9fafb;
        }

        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }

        .message.user {
            text-align: right;
        }

        .message.assistant {
            text-align: left;
        }

        .message-bubble {
            display: inline-block;
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 20px;
            word-wrap: break-word;
        }

        .message.user .message-bubble {
            background: var(--primary-color);
            color: white;
            margin-left: 20%;
        }

        .message.assistant .message-bubble {
            background: white;
            border: 1px solid var(--border-color);
            margin-right: 20%;
        }

        .message-time {
            font-size: 0.8rem;
            color: var(--text-light);
            margin-top: 5px;
        }

        .sources-info {
            margin-top: 10px;
            padding: 10px;
            background: #f0f9ff;
            border-radius: 10px;
            border-left: 4px solid var(--primary-color);
        }

        .sources-info h4 {
            color: var(--primary-color);
            margin-bottom: 5px;
            font-size: 0.9rem;
        }

        .source-item {
            font-size: 0.8rem;
            color: var(--text-light);
            margin-bottom: 5px;
        }

        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid var(--border-color);
        }

        .input-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .question-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid var(--border-color);
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }

        .question-input:focus {
            border-color: var(--primary-color);
        }

        .send-btn {
            padding: 15px 25px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .send-btn:hover:not(:disabled) {
            background: var(--primary-dark);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .quick-questions {
            margin-bottom: 20px;
        }

        .quick-questions h3 {
            color: white;
            margin-bottom: 15px;
            text-align: center;
        }

        .quick-question-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }

        .quick-question-btn {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 12px 16px;
            border-radius: 12px;
            cursor: pointer;
            text-align: left;
            transition: all 0.3s;
            backdrop-filter: blur(10px);
        }

        .quick-question-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
        }

        .control-btn {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            backdrop-filter: blur(10px);
        }

        .control-btn:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .welcome-message {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-light);
        }

        .welcome-message i {
            font-size: 3rem;
            color: var(--primary-color);
            margin-bottom: 20px;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .message-bubble {
                max-width: 90%;
            }

            .quick-question-grid {
                grid-template-columns: 1fr;
            }

            .controls {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-graduation-cap"></i> Pathways Academy</h1>
            <p>Student Handbook Assistant - Ask me anything about school policies, procedures, and guidelines</p>
        </div>

        <div class="quick-questions">
            <h3><i class="fas fa-lightbulb"></i> Quick Questions</h3>
            <div class="quick-question-grid">
                <button class="quick-question-btn" onclick="askQuickQuestion('What is the school\\'s mission and vision?')">
                    <i class="fas fa-bullseye"></i> Mission & Vision
                </button>
                <button class="quick-question-btn" onclick="askQuickQuestion('Tell me about the dress code policy')">
                    <i class="fas fa-tshirt"></i> Dress Code
                </button>
                <button class="quick-question-btn" onclick="askQuickQuestion('What are the attendance requirements?')">
                    <i class="fas fa-calendar-check"></i> Attendance Policy
                </button>
                <button class="quick-question-btn" onclick="askQuickQuestion('How do I contact the school administration?')">
                    <i class="fas fa-phone"></i> Contact Information
                </button>
                <button class="quick-question-btn" onclick="askQuickQuestion('What are the emergency procedures?')">
                    <i class="fas fa-shield-alt"></i> Emergency Procedures
                </button>
                <button class="quick-question-btn" onclick="askQuickQuestion('Tell me about academic policies')">
                    <i class="fas fa-book"></i> Academic Policies
                </button>
            </div>
        </div>

        <div class="chat-container">
            <div class="chat-header">
                <h2><i class="fas fa-robot"></i> Handbook Assistant</h2>
                <p>Powered by AI • Real-time handbook search</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    <i class="fas fa-comments"></i>
                    <h3>Welcome to the Pathways Academy Handbook Assistant!</h3>
                    <p>I'm here to help you find information from the student handbook. You can ask me about school policies, procedures, contact information, and much more.</p>
                    <p><strong>Try asking:</strong> "What are the school hours?" or "How do I report an absence?"</p>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Searching handbook...</p>
            </div>

            <div class="input-container">
                <div class="input-group">
                    <input type="text" 
                           class="question-input" 
                           id="questionInput" 
                           placeholder="Ask me about school policies, procedures, or any handbook topic..."
                           onkeypress="handleKeyPress(event)">
                    <button class="send-btn" id="sendBtn" onclick="askQuestion()">
                        <i class="fas fa-paper-plane"></i>
                        Send
                    </button>
                </div>
            </div>
        </div>

        <div class="controls">
            <button class="control-btn" onclick="clearHistory()">
                <i class="fas fa-trash"></i> Clear Chat
            </button>
            <button class="control-btn" onclick="showHistory()">
                <i class="fas fa-history"></i> View History
            </button>
        </div>
    </div>

    <script>
        let isLoading = false;

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !isLoading) {
                askQuestion();
            }
        }

        function askQuickQuestion(question) {
            document.getElementById('questionInput').value = question;
            askQuestion();
        }

        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            
            if (!question || isLoading) return;
            
            isLoading = true;
            const sendBtn = document.getElementById('sendBtn');
            const loading = document.getElementById('loading');
            
            sendBtn.disabled = true;
            loading.classList.add('show');
            
            // Add user message
            addMessage(question, 'user');
            input.value = '';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(data.response, 'assistant', data.sources_info, data.timestamp);
                } else {
                    addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'), 'assistant');
                }
            } catch (error) {
                addMessage('Sorry, I\\'m having trouble connecting. Please try again.', 'assistant');
                console.error('Error:', error);
            } finally {
                isLoading = false;
                sendBtn.disabled = false;
                loading.classList.remove('show');
                scrollToBottom();
            }
        }

        function addMessage(text, sender, sources = null, timestamp = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const welcomeMessage = messagesContainer.querySelector('.welcome-message');
            
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const currentTime = timestamp || new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            let sourcesHtml = '';
            if (sources && sources.length > 0) {
                sourcesHtml = `
                    <div class="sources-info">
                        <h4><i class="fas fa-search"></i> Sources Found (${sources.length})</h4>
                        ${sources.map((source, index) => 
                            `<div class="source-item">
                                <strong>Source ${index + 1}:</strong> ${source.preview}
                            </div>`
                        ).join('')}
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `
                <div class="message-bubble">
                    ${text.replace(/\\n/g, '<br>')}
                    ${sourcesHtml}
                </div>
                <div class="message-time">${currentTime}</div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            scrollToBottom();
        }

        function scrollToBottom() {
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        async function clearHistory() {
            if (confirm('Are you sure you want to clear the chat history?')) {
                try {
                    await fetch('/clear_history', { method: 'POST' });
                    const messagesContainer = document.getElementById('chatMessages');
                    messagesContainer.innerHTML = `
                        <div class="welcome-message">
                            <i class="fas fa-comments"></i>
                            <h3>Chat history cleared!</h3>
                            <p>Feel free to ask me anything about the student handbook.</p>
                        </div>
                    `;
                } catch (error) {
                    alert('Error clearing history. Please try again.');
                }
            }
        }

        async function showHistory() {
            try {
                const response = await fetch('/history');
                const data = await response.json();
                
                if (data.history && data.history.length > 0) {
                    let historyText = `Chat History (${data.history.length} conversations):\\n\\n`;
                    data.history.forEach((item, index) => {
                        const date = new Date(item.timestamp).toLocaleString();
                        historyText += `${index + 1}. [${date}]\\nQ: ${item.question}\\nA: ${item.response.substring(0, 100)}...\\n\\n`;
                    });
                    alert(historyText);
                } else {
                    alert('No chat history available.');
                }
            } catch (error) {
                alert('Error retrieving history. Please try again.');
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('questionInput').focus();
        });
    </script>
</body>
</html>'''
    
    # Write the template
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_template)

if __name__ == '__main__':
    print("🚀 Starting Pathways Academy Handbook Assistant...")
    
    # Create templates
    create_templates()
    print("📁 Created HTML templates")
    
    # Load RAG data
    if load_rag_data():
        print("🤖 RAG system initialized successfully")
        print("🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load RAG data. Please ensure FAISS index and chunks files exist.")
        print("Expected files:")
        print(f"  - {FAISS_INDEX_PATH}")
        print(f"  - {CHUNKS_DATA_PATH}")