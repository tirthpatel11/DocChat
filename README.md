# DocChat

A powerful document Q&A application that uses AI to answer questions from your PDFs, images, and text files. Built with a FastAPI backend and a React chat interface.

## Features

- **Multi-format Support** — Upload PDFs, images (JPG, PNG), and text files (TXT, MD)
- **Table Extraction** — Automatically extracts and indexes tables from PDFs for structured data Q&A
- **Smart Chunking** — 800-token chunks with 200-token overlap and metadata (page numbers, section titles)
- **Source Citations** — See exactly which document pages and sections were used, with relevance scores
- **Confidence Scoring** — Each answer is tagged with a confidence level (high / medium / low)
- **Chunk Deduplication** — Overlapping chunks are deduplicated so retrieval slots aren't wasted
- **Markdown Answers** — Responses are rendered as Markdown with tables, lists, and code blocks
- **Chat Interface** — Conversational UI with full message history per session

## Tech Stack

- **Backend**: FastAPI, LangChain, Groq LLM, ObjectBox Vector Store
- **Frontend**: React 18, Vite, Axios, react-markdown
- **AI Models**:
  - LLM: Llama 3.3 70B Versatile (via Groq)
  - Embeddings: BAAI/bge-base-en-v1.5 (768-d)

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- Groq API key ([Get one here](https://console.groq.com/))

### Backend Setup

1. Install Python dependencies:
```bash
cd app
pip install -r ../requirements.txt
```

2. Create `.env` file in `app/` directory:
```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

3. Start the backend server:
```bash
python3 -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

1. Install Node dependencies:
```bash
cd frontend
npm install
```

2. (Optional) Create `.env` file if backend is on a different URL:
```bash
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
```

3. Start the development server:
```bash
npm run dev
```

4. Open `http://localhost:5173` in your browser

## Usage

1. Upload your PDF, image, or text documents using the sidebar
2. Click **Embed documents** to index them
3. Ask questions in the chat input
4. View answers (rendered as Markdown) with source citations and relevance scores

## Project Structure

```
DocChat-main/
├── app/
│   ├── backend.py          # FastAPI backend — RAG pipeline, vector store, query endpoint
│   ├── utils.py            # LLM and embedding model configuration
│   ├── config.py           # Environment variable management
│   └── project/            # ObjectBox vector database (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Chat interface with message history and markdown rendering
│   │   ├── main.jsx        # React entry point
│   │   └── styles.css      # Dark-theme chat layout styles
│   ├── index.html
│   └── package.json
└── requirements.txt        # Python dependencies
```
