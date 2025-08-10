<img width="1905" height="864" alt="Screenshot 2025-08-10 194047" src="https://github.com/user-attachments/assets/460b2200-6568-4c1c-80d8-75ce8059a212" />
# Contextual RAG Chatbot (Gemini 2.5)

An advanced, Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that analyzes multiple PDF documents and answers questions with grounded, context-aware responses powered by Google's Gemini 2.5.

## Features

- Multi-document PDF ingestion (up to 100MB each)
- Fast chunking + embedding-based retrieval
- Contextual conversation memory with configurable context window
- Gemini 2.5 responses with streaming output and markdown formatting
- Source references for retrieved passages
- Advanced UI with avatars, status chips, and analytics dashboard
- Secure API key management via `.env`

## Demo (What You’ll See)

- Left sidebar: configuration (model parameters, chunking settings) and document upload
- Main chat area: user/assistant messages with avatars and streaming responses
- Right panel: document overview and analytics (chunks per document, file size distribution, chunk size histogram)

## Requirements

- Python 3.9 – 3.11 recommended
- Google Gemini API key

## Quick Start

1) Clone this repository and enter the project directory.

2) Create a virtual environment and activate it:
```
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

3) Install dependencies:
```
pip install -r requirements.txt
```

4) Create a `.env` file in the project root with your Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5) Run the app:
```
streamlit run app.py
```

Open the provided local URL in your browser.

## Usage

- In the sidebar, upload one or more PDF files (each ≤ 100MB).
- Click “Process Documents” to extract text, chunk, and embed.
- Ask questions in the chat input. The assistant streams its answer.
- Expand “Source References” below an answer to see relevant passages grouped by document.

## Configuration

- Temperature: 0.0–1.0 (creativity vs determinism)
- Context Window: how many past turns to consider
- Chunk Size / Overlap: controls chunking granularity and continuity
- Top K Results: how many most relevant chunks feed RAG

These are set in `app.py` and saved in Streamlit session state.

## How It Works (Architecture)

- PDF ingestion: `PyPDF2` reads pages in batches with progress tracking
- Chunking: sentence-aware splitting with configurable size/overlap
- Embeddings: `google-generativeai` embedding model `models/embedding-001`
- Retrieval: cosine similarity over in-session vector store
- Generation: `gemini-2.5-pro` with a RAG-oriented system instruction, markdown responses, and streaming
- UI: Streamlit chat with avatars, analytics using Plotly

## Environment and Security

- The API key is loaded from `.env` using `python-dotenv` and never shown in the UI.
- The app will warn in the sidebar if `GEMINI_API_KEY` is missing and disable processing/chat.

## Troubleshooting

- Streaming error / empty response:
  - Ensure your key is valid and has access to Gemini 2.5.
  - Re-run with a fresh environment; check network stability.
- PDF parsing issues:
  - Some scanned PDFs may have no extractable text. Consider OCR preprocessing.
- Memory / large loads:
  - Reduce `chunk_size`, `top_k_results`, or process fewer PDFs at a time.

## Extending the App

- Swap vector store: integrate FAISS/Chroma/Pinecone for persistence at scale.
- Add observability: integrate LangSmith/TruLens for evaluation and tracing.
- Add authentication: gate the UI with Streamlit auth or a reverse proxy.

## Project Structure

```
rag-chatbot/
  app.py               # Main Streamlit app
  requirements.txt     # Python dependencies
  venv/                # Virtual environment (optional)
```

## License

This project is provided as-is, without warranties. Adapt and extend for your use case.

