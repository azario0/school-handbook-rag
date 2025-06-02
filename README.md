# 🎓 Pathways Academy - Student Handbook RAG System

A modern, AI-powered web application that provides intelligent question-answering capabilities for school handbook content using Retrieval-Augmented Generation (RAG) technology.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **🤖 AI-Powered Q&A**: Uses Google's Gemini AI for intelligent responses
- **🔍 Vector Search**: FAISS-based semantic search for accurate information retrieval
- **💻 Modern Web Interface**: Responsive, mobile-friendly design with chat interface
- **⚡ Real-time Processing**: Instant answers with loading indicators
- **📱 Mobile Responsive**: Works seamlessly on all devices
- **💾 Session Management**: Maintains chat history during sessions
- **🎯 Quick Questions**: Pre-defined common questions for easy access

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **AI/ML**: Google Gemini API, FAISS vector database
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Embeddings**: Google's text-embedding-004 model
- **Vector Search**: Facebook AI Similarity Search (FAISS)

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Pre-processed school handbook data (FAISS index and chunks)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/azario0/school-handbook-rag.git
   cd school-handbook-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   echo "SECRET_KEY=your_secret_key_here" >> .env
   ```

5. **Ensure data files exist**
   Make sure these files are in your project directory:
   - `school_handbook.faiss` (FAISS vector index)
   - `school_handbook_chunks.pkl` (Text chunks data)

## 📦 Requirements

Create a `requirements.txt` file with:

```
Flask==2.3.3
google-generativeai==0.3.2
faiss-cpu==1.7.4
numpy==1.24.3
python-dotenv==1.0.0
```

## 🚀 Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Start asking questions!**
   - Use the quick question buttons for common queries
   - Type your own questions about school policies, procedures, etc.
   - View chat history and clear conversations as needed

## 📁 Project Structure

```
school-handbook-rag/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
│
├── templates/
│   └── index.html                 # Auto-generated HTML template
│
├── static/                        # Static files (auto-created)
│
├── school_handbook.faiss          # FAISS vector index (required)
├── school_handbook_chunks.pkl     # Text chunks data (required)
│
└── notebook2.ipynb               # Original Jupyter notebook
```

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (defaults provided)
SECRET_KEY=your_flask_secret_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/ask` | POST | Submit questions and get AI responses |
| `/history` | GET | Retrieve chat history |
| `/clear_history` | POST | Clear chat history |
| `/health` | GET | System health check |

## 📊 Data Preparation

To prepare your own handbook data:

1. **Text Processing**: Extract and clean text from your handbook
2. **Chunking**: Split text into meaningful chunks (see `notebook2.ipynb`)
3. **Embedding**: Generate embeddings using Gemini's text-embedding-004
4. **Indexing**: Create FAISS index for vector similarity search

Example using the provided notebook:
```python
# Follow the steps in notebook2.ipynb to:
# 1. Load and chunk your handbook text
# 2. Generate embeddings
# 3. Create FAISS index
# 4. Save the processed data
```

## 🎨 Customization

### Styling
- Modify CSS variables in the HTML template for color schemes
- Update the header and branding information
- Customize quick question buttons

### Functionality
- Adjust `top_k` parameter for number of retrieved chunks
- Modify the response generation prompt for different tones
- Add new API endpoints for additional features

## 🚦 Production Deployment

For production deployment:

1. **Use environment variables for sensitive data**
2. **Use a production WSGI server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```
3. **Set up reverse proxy (nginx)**
4. **Enable HTTPS**
5. **Implement rate limiting**
6. **Add logging and monitoring**

## 🧪 Testing

```bash
# Test the health endpoint
curl http://localhost:5000/health

# Test question asking (replace with actual question)
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the school mission?"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini AI for powerful language model capabilities
- Facebook AI Research for FAISS vector search
- Flask community for the excellent web framework
- Contributors and testers who helped improve this system

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/azario0/school-handbook-rag/issues) page
2. Create a new issue with detailed description
3. Contact the development team

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output capabilities
- [ ] Advanced analytics and usage tracking
- [ ] Integration with school management systems
- [ ] Mobile app development
- [ ] Advanced caching mechanisms
- [ ] User authentication and personalization

---

**Made with ❤️ for educational excellence**

*This project demonstrates the power of AI in making educational resources more accessible and interactive.*
