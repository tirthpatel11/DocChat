import React, { useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [files, setFiles] = useState([]);
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState("");
  const [sources, setSources] = useState([]);
  const [isAsking, setIsAsking] = useState(false);

  const handleFileChange = (event) => {
    setFiles(Array.from(event.target.files || []));
    setIsReady(false);
    setAnswer("");
    setError("");
  };

  const handleEmbed = async () => {
    if (!files.length) {
      setError("Please select at least one PDF or image file.");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setIsEmbedding(true);
    setError("");

    try {
      await axios.post(`${API_BASE_URL}/embed`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setIsReady(true);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to embed documents.");
      setIsReady(false);
    } finally {
      setIsEmbedding(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    setError("");
    setAnswer("");
    setResponseTime(null);
    setSources([]);
    setIsAsking(true);

    try {
      const res = await axios.post(`${API_BASE_URL}/query`, {
        question: question.trim(),
      });
      setAnswer(res.data.answer);
      setResponseTime(res.data.response_time);
      setSources(res.data.sources || []);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to query documents.");
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <span className="logo-dot" />
          <span className="logo-text">DocChat</span>
        </div>
        <div className="header-right">
          <span className="badge">AI Document Q&A</span>
        </div>
      </header>

      <main className="main">
        <section className="panel upload-panel">
          <h2>Upload your documents</h2>
          <p className="subtitle">
            Add PDFs or images (JPG, PNG). We&apos;ll index them so you can ask natural
            language questions.
          </p>

          <div className="upload-box">
            <input
              id="file-input"
              type="file"
              accept=".pdf,image/jpeg,image/png"
              multiple
              onChange={handleFileChange}
            />
            <label htmlFor="file-input" className="upload-label">
              <span className="upload-icon">⬆</span>
              <span>
                <strong>Click to upload</strong> or drag and drop
              </span>
              <span className="upload-hint">PDF, JPG, PNG up to your server limits</span>
            </label>
          </div>

          {files.length > 0 && (
            <div className="file-list">
              {files.map((file) => (
                <span key={file.name} className="file-chip">
                  {file.name}
                </span>
              ))}
            </div>
          )}

          <button
            className="primary-button"
            onClick={handleEmbed}
            disabled={isEmbedding || !files.length}
          >
            {isEmbedding ? "Embedding…" : "Embed documents"}
          </button>

          {isReady && (
            <p className="status success">
              ✅ Documents embedded. You can now ask questions.
            </p>
          )}
          {error && <p className="status error">⚠ {error}</p>}
        </section>

        <section className="panel qa-panel">
          <h2>Ask your documents</h2>
          <p className="subtitle">
            Once your documents are embedded, ask any question and DocChat will answer
            using their content.
          </p>

          <textarea
            className="question-input"
            rows={4}
            placeholder="e.g. What are the key takeaways from the contract?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={!isReady}
          />

          <div className="actions-row">
            <button
              className="secondary-button"
              onClick={handleAsk}
              disabled={!isReady || isAsking}
            >
              {isAsking ? "Thinking…" : "Ask question"}
            </button>
            {!isReady && (
              <span className="hint">Upload & embed documents first to enable Q&A.</span>
            )}
          </div>

          <div className="answer-box">
            <h3>Answer</h3>
            {answer ? (
              <p className="answer-text">{answer}</p>
            ) : (
              <p className="placeholder">
                Your answer will appear here once you ask a question.
              </p>
            )}
            {responseTime !== null && (
              <p className="meta">
                Response time: <strong>{responseTime.toFixed(2)}s</strong>
              </p>
            )}

            {sources.length > 0 && (
              <div className="sources">
                <h4>Sources</h4>
                <ul>
                  {sources.slice(0, 4).map((src, idx) => (
                    <li key={idx}>
                      <div className="source-meta">
                        <span className="source-name">
                          {src.source || "Document snippet"}
                        </span>
                        {src.page !== null && src.page !== undefined && (
                          <span className="source-page">Page {src.page + 1}</span>
                        )}
                      </div>
                      <p className="source-snippet">{src.content}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="footer">
        <span>DocChat • Built with FastAPI and React</span>
      </footer>
    </div>
  );
}

export default App;



