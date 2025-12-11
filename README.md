# DocChat

A powerful document Q&A application that uses AI to answer questions from your PDFs and images. Built with FastAPI backend and React frontend.

## Features

- 📄 **Multi-format Support**: Upload PDFs and images (JPG, PNG)
- 🔍 **Table Extraction**: Automatically extracts and indexes tables from PDFs for structured data Q&A
- 🧠 **Smart Chunking**: Advanced chunking with metadata (page numbers, section titles, headings)
- 📊 **Source Citations**: See exactly which document pages and sections were used to answer your questions
- 🎯 **Confidence Scoring**: Low-confidence answers show raw snippets instead of hallucinating
- ⚡ **Fast Retrieval**: Optimized vector search with similarity thresholds

## Tech Stack

- **Backend**: FastAPI, LangChain, Groq LLM, ObjectBox Vector Store
- **Frontend**: React, Vite, Axios
- **AI Models**: 
  - LLM: Llama 3.1 8B Instant (via Groq)
  - Embeddings: BAAI/bge-small-en-v1.5

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

2. (Optional) Create `.env` file if backend is on different URL:
```bash
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
```

3. Start the development server:
```bash
npm run dev
```

4. Open `http://localhost:5173` in your browser

## Usage

1. Upload your PDF or image documents
2. Click "Embed documents" to index them
3. Ask questions about your documents
4. View answers with source citations showing which pages were used

## Project Structure

```
DocChat-main/
├── app/
│   ├── backend.py          # FastAPI backend server
│   ├── utils.py            # LLM and embedding utilities
│   ├── config.py           # Configuration management
│   └── project/            # ObjectBox vector database
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── main.jsx        # Entry point
│   │   └── styles.css      # Styling
│   └── package.json
└── requirements.txt        # Python dependencies
```

## License

MIT

