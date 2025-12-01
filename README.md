# 🤖 Contextual RAG Chatbot

> An intelligent document assistant powered by Google Gemini AI, featuring advanced RAG capabilities, contextual conversations, and comprehensive analytics.

![RAG Chatbot Banner](https://github.com/user-attachments/assets/6dc6c9db-e419-4e44-9ae2-bacd4ade3b2c)

## ✨ Key Features

- **🧠 Advanced AI**: Powered by Google Gemini with thinking & reflection capabilities
- **📚 Smart Document Processing**: PDF processing with multiple extraction methods and OCR fallback
- **🔍 Hybrid Search**: Combines semantic and keyword search for optimal retrieval
- **💬 Contextual Conversations**: Maintains conversation context with adjustable window (1-20 messages)
- **📊 Analytics Dashboard**: Real-time metrics, interactive charts, and Excel/HTML export
- **🎨 Modern UI**: 4 themes, responsive design, accessibility features
- **🔒 Enterprise Security**: Privacy mode, input validation, session isolation
- **⚡ High Performance**: Async processing, intelligent caching, circuit breakers

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 8GB RAM (16GB recommended)
- Google Gemini API key

### Installation

```bash
# Clone repository
git clone https://github.com/Anupam0202/Contextual-RAG-Chatbot.git
cd Contextual-RAG-Chatbot

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

1. **Create `.env` file:**
```bash
cp .env.example .env
```

2. **Add your Gemini API key:**
```env
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Run

```bash
streamlit run app.py
```

Application opens at `http://localhost:8501`

## 📖 How to Use

### 1. Upload Documents
- Navigate to **📚 Documents** page
- Upload PDF files (drag & drop or browse)
- Wait for processing to complete

### 2. Chat with Your Documents
- Go to **💬 Chat** page
- Ask questions about your documents
- Get AI responses with source citations

**Example queries:**
- "What are the main topics in this document?"
- "Summarize the key findings"
- "Compare section 2 and section 5"

### 3. View Analytics
- Visit **📊 Analytics** page
- Review performance metrics
- Export reports (Excel/HTML)

### 4. Adjust Settings
- Open **⚙️ Settings** page
- Configure model parameters
- Adjust context window
- Set UI preferences

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         Streamlit UI (app.py)       │
│   [Chat | Documents | Analytics]    │
├─────────────────────────────────────┤
│      RAG Engine (rag_core.py)       │
│ [Planning→Retrieval→Generation]     │
├──────────────┬──────────────────────┤
│ Vector Store │   PDF Processor      │
│ • Hybrid     │   • Chunking         │
│ • FAISS      │   • OCR Fallback     │
├──────────────┴──────────────────────┤
│   Infrastructure (utils, config)    │
│     [Cache | Sessions | Security]   │
└─────────────────────────────────────┘
```

### Query Processing Flow

```
User Query → Planning → Vector Search → Reranking → 
Generation → Reflection → Response
```

## ⚙️ Configuration Options

### Model Settings
```env
RAG_MODEL_NAME=gemini-1.5-flash      # AI model
RAG_TEMPERATURE=0.7                   # Creativity (0-1)
RAG_MAX_OUTPUT_TOKENS=2048           # Max response length
```

### Retrieval Settings
```env
RAG_RETRIEVAL_TOP_K=5                # Results per query
RAG_CHUNK_SIZE=1000                  # Text chunk size
RAG_CHUNK_OVERLAP=200                # Chunk overlap
RAG_HYBRID_SEARCH_ALPHA=0.5          # Search balance
```

### Performance Settings
```env
RAG_ENABLE_CACHING=true              # Enable caching
RAG_CACHE_TTL=3600                   # Cache lifetime (seconds)
RAG_MAX_WORKERS=4                    # Parallel workers
```

## 🔧 Troubleshooting

### Common Issues

**Application won't start**
```bash
# Ensure venv is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**API Key Error**
```bash
# Verify .env file exists with correct key
cat .env  # macOS/Linux
type .env # Windows
```

**PDF Processing Fails**
- Ensure PDF isn't corrupted
- Check file size (\u003c50MB)
- For scanned PDFs, install Tesseract OCR

**Memory Issues**
- Reduce chunk size in settings
- Limit context window to 3-5 messages
- Clear cache regularly

### Debug Mode

```bash
# Enable debug logging
streamlit run app.py --logger.level=debug
```

## 📚 API Usage

### Basic Example

```python
from rag_core import getRAGEngine
from pdf_processor import createPDFProcessor

# Process PDF
processor = createPDFProcessor()
doc = processor.processPDF("document.pdf")

# Query documents
rag = getRAGEngine()
async for chunk in rag.processQuery("What is the summary?"):
    print(chunk, end="")
```

### With Conversation History

```python
conversation = [
    {"role": "user", "content": "What is this about?"},
    {"role": "assistant", "content": "It's about..."}
]

async for response in rag.processQuery(
    "Tell me more", 
    conversation_history=conversation
):
    print(response, end="")
```

## 🎯 Advanced Features

### Thinking & Reflection
Multi-step reasoning process for complex queries:
- **Planning**: Intent classification & query decomposition
- **Retrieval**: Hybrid search with reranking
- **Generation**: Context-aware responses
- **Reflection**: Self-evaluation & improvement

### Circuit Breaker
Automatic failure recovery prevents cascade failures:
- `CLOSED` → normal operation
- `OPEN` → failures detected, requests blocked
- `HALF_OPEN` → testing recovery
- Back to `CLOSED` → recovered

### Privacy Features
- PII sanitization
- Sensitive data redaction
- Session timeout management
- Secure API key storage

## ❓ FAQ

**Q: What file formats are supported?**  
A: Currently PDF files (.pdf). Text file support coming soon.

**Q: What's the file size limit?**  
A: Default is 50MB, configurable in settings.

**Q: Is my data secure?**  
A: Yes - local processing, privacy mode, session isolation, secure API management.

**Q: Which AI models are supported?**  
A: Google Gemini 1.5 Flash (default), Gemini 1.5 Pro, Gemini 1.0 Pro

**Q: How does hybrid search work?**  
A: Combines semantic search (meaning) + keyword search (exact matches) with configurable weighting.

**Q: What is context window?**  
A: Number of previous messages (1-20) included when generating responses.

## 🛠️ Tech Stack

- **AI**: Google Gemini 1.5
- **Framework**: Streamlit
- **Embeddings**: Sentence Transformers
- **Vector DB**: FAISS
- **PDF**: PyPDF2, pdfplumber, Tesseract OCR
- **Analytics**: Plotly, Pandas

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Anupam0202/Contextual-RAG-Chatbot/issues)
- **Documentation**: [GitHub README](https://github.com/Anupam0202/Contextual-RAG-Chatbot)

## 🙏 Acknowledgments

Built with:
- [Google Gemini AI](https://ai.google/gemini/) - AI model
- [Streamlit](https://streamlit.io/) - Web framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

---

**Made with ❤️ by Anupam**
