import { useState, useRef, useEffect } from "react";
import { predictReview } from "./api";
import { SentimentBadge } from "./components/SentimentBadge";
import "./styles.css";

function prettyPct(p) {
  if (typeof p !== "number") return "—";
  return `${(p * 100).toFixed(1)}%`;
}

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [rephrase, setRephrase] = useState("");
  const taRef = useRef(null);

  // Set page title
  useEffect(() => {
    document.title = "Rent Runway - Feedback";
  }, []);

  async function analyze() {
    setError("");
    setResult(null);
    if (!text.trim()) return setError("Please paste or type a review.");
    try {
      setLoading(true);
      const data = await predictReview(text.trim());
      if (!data || typeof data !== "object") throw new Error("Bad API response");
      setResult(data);
      setRephrase(data.rephrase || "");
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const copy = async (v) => {
    try { await navigator.clipboard.writeText(v); }
    catch { setError("Copy failed. Use Ctrl+C."); }
  };

  const downloadJSON = () => {
    if (!result) return;
    const payload = { review_text: text, ...result, rephrase, exported_at: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "review_analysis.json"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <header className="header">
        <div className="hero">
          <h1 className="fancy-header"> Style Speaks — Tell Us What You Think</h1>
          <p className="sub">
            Paste a product review—get sentiment, reasons, a detailed explanation, and a brand-friendly rephrase.
          </p>
        </div>
      </header>

      {/* Input */}
      <section className="card">
        <label className="label">Review</label>
        <textarea
          ref={taRef}
          className="textarea"
          rows={6}
          placeholder="Paste or type the customer review here…"
          value={text}
          onChange={(e)=>setText(e.target.value)}
          onKeyDown={(e)=>{ if(e.ctrlKey && e.key==="Enter") analyze(); }}
        />
        <div className="row">
          <button className="btn primary" onClick={analyze} disabled={loading}>
            {loading ? "Analyzing…" : "Analyze (Ctrl+Enter)"}
          </button>
          <button className="btn" onClick={()=>{
            setText(""); setResult(null); setRephrase(""); setError(""); taRef.current?.focus();
          }} disabled={loading}>
            Clear
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {/* Results */}
      {result && (
        <section className="card glass">
          <div className="row wrap gap center" style={{ marginTop: 12 }}>
            <SentimentBadge sentiment={result.sentiment} />
            <div className="prob">{prettyPct(result.probability)} confidence</div>
          </div>

          <div className="block">
            <div className="label">Reason</div>
            <div className="reason">{result.reason || "—"}</div>
          </div>

          <div className="block">
            <div className="label">Detailed Explanation</div>
            <div className="panel">{result.explanation || "—"}</div>
            <div className="row">
              <button className="btn" onClick={()=>copy(result.explanation || "")}>Copy Explanation</button>
            </div>
          </div>

          <div className="block">
            <div className="label">Rephrase</div>
            <textarea className="textarea" rows={4} value={rephrase} onChange={(e)=>setRephrase(e.target.value)} />
            <div className="row">
              <button className="btn" onClick={()=>copy(rephrase)}>Copy Rephrase</button>
              <button className="btn" onClick={()=>copy(JSON.stringify({ ...result, review_text:text, rephrase }, null, 2))}>Copy JSON</button>
              <button className="btn outline" onClick={downloadJSON}>Export (Download JSON)</button>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
