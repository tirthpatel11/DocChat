import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [files, setFiles] = useState([]);
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [docCount, setDocCount] = useState(0);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState("");
  const [isAsking, setIsAsking] = useState(false);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isAsking]);

  const handleFileChange = (event) => {
    setFiles(Array.from(event.target.files || []));
    setError("");
  };

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleEmbed = async () => {
    if (!files.length) {
      setError("Please select at least one file.");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setIsEmbedding(true);
    setError("");

    try {
      const res = await axios.post(`${API_BASE_URL}/embed`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setIsReady(true);
      setDocCount(res.data.document_chunks || 0);
      setMessages([]);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to embed documents.");
      setIsReady(false);
    } finally {
      setIsEmbedding(false);
    }
  };

  const handleAsk = async () => {
    const q = question.trim();
    if (!q || !isReady || isAsking) return;

    setError("");
    setQuestion("");
    setIsAsking(true);

    setMessages((prev) => [...prev, { role: "user", content: q }]);

    try {
      const res = await axios.post(`${API_BASE_URL}/query`, { question: q });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.data.answer,
          sources: res.data.sources || [],
          responseTime: res.data.response_time,
          confidence: res.data.confidence,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, an error occurred while processing your question.",
          isError: true,
        },
      ]);
      setError(err.response?.data?.detail || "");
    } finally {
      setIsAsking(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const fileIcon = (name) => {
    if (name.endsWith(".pdf")) return "pdf";
    if (/\.(jpe?g|png)$/i.test(name)) return "img";
    return "txt";
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="logo-dot" />
          <span className="logo-text">DocChat</span>
        </div>

        <div className="sidebar-body">
          <div className="upload-section">
            <h3 className="section-title">Documents</h3>
            <p className="section-hint">
              Upload PDFs, images, or text files to start asking questions.
            </p>

            <div className="upload-drop">
              <input
                id="file-input"
                type="file"
                accept=".pdf,.txt,.md,image/jpeg,image/png"
                multiple
                onChange={handleFileChange}
              />
              <label htmlFor="file-input" className="drop-label">
                <span className="drop-icon">+</span>
                <span className="drop-text">Choose files</span>
              </label>
            </div>

            {files.length > 0 && (
              <ul className="file-list">
                {files.map((file) => (
                  <li key={file.name} className="file-item">
                    <span className={`file-badge ${fileIcon(file.name)}`}>
                      {fileIcon(file.name).toUpperCase()}
                    </span>
                    <span className="file-name">{file.name}</span>
                    <button
                      className="file-remove"
                      onClick={() => removeFile(file.name)}
                      aria-label={`Remove ${file.name}`}
                    >
                      &times;
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <button
              className="btn-embed"
              onClick={handleEmbed}
              disabled={isEmbedding || !files.length}
            >
              {isEmbedding ? (
                <>
                  <span className="spinner" /> Processing...
                </>
              ) : (
                "Embed documents"
              )}
            </button>
          </div>

          {isReady && (
            <div className="status-pill success">
              <span className="status-dot" />
              Ready &mdash; {docCount} chunks indexed
            </div>
          )}
          {error && <div className="status-pill error">{error}</div>}
        </div>

        <div className="sidebar-footer">
          <span>Built with FastAPI &amp; React</span>
        </div>
      </aside>

      <main className="chat">
        <div className="chat-scroll">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-glyph">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                </svg>
              </div>
              <h2 className="empty-title">Ask your documents anything</h2>
              <p className="empty-sub">
                {isReady
                  ? "Your documents are indexed and ready. Type a question below."
                  : "Upload and embed documents from the sidebar to get started."}
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`msg msg-${msg.role}`}>
              <div className="msg-avatar">
                {msg.role === "user" ? "Y" : "D"}
              </div>
              <div className="msg-body">
                {msg.role === "assistant" ? (
                  <>
                    <div
                      className={`msg-content markdown ${msg.isError ? "msg-error" : ""}`}
                    >
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                    {msg.responseTime !== undefined && (
                      <div className="msg-meta">
                        <span className="meta-time">
                          {msg.responseTime.toFixed(2)}s
                        </span>
                        {msg.confidence && (
                          <span
                            className={`meta-confidence conf-${msg.confidence}`}
                          >
                            {msg.confidence}
                          </span>
                        )}
                      </div>
                    )}
                    {msg.sources?.length > 0 && (
                      <SourceList sources={msg.sources} />
                    )}
                  </>
                ) : (
                  <div className="msg-content">
                    <p>{msg.content}</p>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isAsking && (
            <div className="msg msg-assistant">
              <div className="msg-avatar">D</div>
              <div className="msg-body">
                <div className="typing">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="chat-bar">
          <textarea
            ref={textareaRef}
            className="chat-input"
            rows={1}
            placeholder={
              isReady
                ? "Ask a question about your documents…"
                : "Upload documents first…"
            }
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!isReady || isAsking}
          />
          <button
            className="btn-send"
            onClick={handleAsk}
            disabled={!isReady || isAsking || !question.trim()}
            aria-label="Send"
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </main>
    </div>
  );
}

function SourceList({ sources }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="sources">
      <button className="sources-btn" onClick={() => setOpen(!open)}>
        <span className="sources-chevron">{open ? "\u25BE" : "\u25B8"}</span>
        {sources.length} source{sources.length !== 1 ? "s" : ""}
      </button>
      {open && (
        <div className="sources-grid">
          {sources.map((src, idx) => (
            <div key={idx} className="src-card">
              <div className="src-head">
                <span className="src-name">{src.source || "Document"}</span>
                {src.page != null && (
                  <span className="src-page">Page {src.page + 1}</span>
                )}
                {src.relevance != null && (
                  <span className="src-rel">
                    {(src.relevance * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <p className="src-text">{src.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
