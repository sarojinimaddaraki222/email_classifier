import { useState, useEffect, useRef } from "react";

const INITIAL_EMAILS = [
  {
    id: 1,
    subject: "URGENT: Payment system completely down",
    body: "Our payment system is not working since 3am. We're losing thousands of dollars per hour. This is a critical emergency.",
    category: "complaint",
    urgency: "high",
    confidence: 0.96,
    sentiment: "negative",
    summary: "Critical payment system outage causing major revenue loss.",
    timestamp: new Date(Date.now() - 2 * 3600000),
    tags: ["outage", "revenue", "critical"],
  },
  {
    id: 2,
    subject: "Feature Request: Dark mode please!",
    body: "Would love to see dark mode added to the dashboard. It would really help during late-night sessions.",
    category: "request",
    urgency: "low",
    confidence: 0.88,
    sentiment: "positive",
    summary: "User requesting dark mode feature for nighttime use.",
    timestamp: new Date(Date.now() - 5 * 3600000),
    tags: ["ui", "feature", "accessibility"],
  },
  {
    id: 3,
    subject: "Amazing customer support experience!",
    body: "Thank you for the excellent support from your team. Sarah was incredibly helpful and resolved my issue in minutes.",
    category: "feedback",
    urgency: "low",
    confidence: 0.94,
    sentiment: "positive",
    summary: "Positive feedback praising quick support resolution.",
    timestamp: new Date(Date.now() - 24 * 3600000),
    tags: ["praise", "support", "team"],
  },
  {
    id: 4,
    subject: "Congratulations! You've won $5,000,000!!",
    body: "Click here now to claim your prize. This offer expires in 24 hours. Act immediately!",
    category: "spam",
    urgency: "low",
    confidence: 0.99,
    sentiment: "neutral",
    summary: "Obvious spam email with lottery scam.",
    timestamp: new Date(Date.now() - 30 * 60000),
    tags: ["scam", "lottery"],
  },
];

const CATEGORY_CONFIG = {
  complaint: { color: "#e24b4a", bg: "#fcebeb", label: "Complaint", icon: "⚠" },
  request: { color: "#185fa5", bg: "#e6f1fb", label: "Request", icon: "✦" },
  feedback: { color: "#3b6d11", bg: "#eaf3de", label: "Feedback", icon: "★" },
  spam: { color: "#5f5e5a", bg: "#f1efe8", label: "Spam", icon: "✕" },
};

const URGENCY_CONFIG = {
  high: { color: "#a32d2d", bg: "#f7c1c1", label: "High", dot: "#e24b4a" },
  medium: { color: "#854f0b", bg: "#fac775", label: "Medium", dot: "#ef9f27" },
  low: { color: "#3b6d11", bg: "#c0dd97", label: "Low", dot: "#639922" },
};

function timeAgo(date) {
  const diff = (Date.now() - date) / 1000;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function MiniBar({ value, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, height: 4, background: "#f1efe8", borderRadius: 99 }}>
        <div
          style={{
            width: `${value * 100}%`,
            height: "100%",
            background: color,
            borderRadius: 99,
            transition: "width 0.6s ease",
          }}
        />
      </div>
      <span style={{ fontSize: 11, color: "#888780", minWidth: 28 }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

function CategoryPill({ category }) {
  const cfg = CATEGORY_CONFIG[category] || CATEGORY_CONFIG.spam;
  return (
    <span
      style={{
        background: cfg.bg,
        color: cfg.color,
        fontSize: 11,
        fontWeight: 500,
        padding: "3px 9px",
        borderRadius: 99,
        letterSpacing: "0.02em",
      }}
    >
      {cfg.icon} {cfg.label}
    </span>
  );
}

function UrgencyDot({ urgency }) {
  const cfg = URGENCY_CONFIG[urgency] || URGENCY_CONFIG.medium;
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: cfg.dot,
          display: "inline-block",
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 11, color: "#5f5e5a" }}>{cfg.label}</span>
    </span>
  );
}

function DonutChart({ data }) {
  const total = data.reduce((s, d) => s + d.count, 0) || 1;
  let offset = 0;
  const r = 54;
  const cx = 70;
  const cy = 70;
  const circ = 2 * Math.PI * r;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
      <svg width={140} height={140} viewBox="0 0 140 140">
        {data.map((d, i) => {
          const pct = d.count / total;
          const dash = pct * circ;
          const gap = circ - dash;
          const rot = (offset / total) * 360 - 90;
          offset += d.count;
          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={r}
              fill="none"
              stroke={d.color}
              strokeWidth={16}
              strokeDasharray={`${dash} ${gap}`}
              style={{ transformOrigin: `${cx}px ${cy}px`, transform: `rotate(${rot}deg)` }}
            />
          );
        })}
        <text x={cx} y={cy - 6} textAnchor="middle" fontSize={22} fontWeight={500} fill="#2c2c2a">
          {total}
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fontSize={10} fill="#888780">
          total
        </text>
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {data.map((d, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: d.color, flexShrink: 0 }} />
            <span style={{ fontSize: 12, color: "#5f5e5a", minWidth: 60 }}>{d.label}</span>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#2c2c2a" }}>{d.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TypewriterText({ text, speed = 12 }) {
  const [displayed, setDisplayed] = useState("");
  const idx = useRef(0);

  useEffect(() => {
    idx.current = 0;
    setDisplayed("");
    if (!text) return;
    const iv = setInterval(() => {
      idx.current++;
      setDisplayed(text.slice(0, idx.current));
      if (idx.current >= text.length) clearInterval(iv);
    }, speed);
    return () => clearInterval(iv);
  }, [text]);

  return <span>{displayed}</span>;
}

function EmailCard({ email, isNew }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = CATEGORY_CONFIG[email.category] || CATEGORY_CONFIG.spam;

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      style={{
        background: "white",
        border: "0.5px solid #d3d1c7",
        borderLeft: `3px solid ${cfg.color}`,
        borderRadius: 10,
        padding: "14px 16px",
        cursor: "pointer",
        transition: "box-shadow 0.15s",
        animation: isNew ? "slideIn 0.35s ease" : undefined,
      }}
      onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.08)")}
      onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "none")}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 13, fontWeight: 500, color: "#2c2c2a", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {email.subject}
            </span>
            {isNew && (
              <span style={{ background: "#e6f1fb", color: "#185fa5", fontSize: 9, fontWeight: 500, padding: "2px 6px", borderRadius: 99, flexShrink: 0 }}>
                NEW
              </span>
            )}
          </div>
          <p style={{ fontSize: 12, color: "#888780", margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: expanded ? "normal" : "nowrap" }}>
            {email.summary}
          </p>
        </div>
        <span style={{ fontSize: 11, color: "#b4b2a9", flexShrink: 0 }}>{timeAgo(email.timestamp)}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
        <CategoryPill category={email.category} />
        <UrgencyDot urgency={email.urgency} />
        <div style={{ flex: 1, maxWidth: 120, marginLeft: "auto" }}>
          <MiniBar value={email.confidence} color={cfg.color} />
        </div>
      </div>

      {expanded && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "0.5px solid #e8e6df" }}>
          <p style={{ fontSize: 12, color: "#5f5e5a", margin: "0 0 10px", lineHeight: 1.6 }}>{email.body}</p>
          {email.tags && (
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {email.tags.map((t) => (
                <span key={t} style={{ fontSize: 10, background: "#f1efe8", color: "#888780", padding: "2px 7px", borderRadius: 99 }}>
                  #{t}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [emails, setEmails] = useState(INITIAL_EMAILS);
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [newId, setNewId] = useState(null);
  const [filterCat, setFilterCat] = useState("all");
  const [filterUrg, setFilterUrg] = useState("all");
  const [search, setSearch] = useState("");
  const [streamText, setStreamText] = useState("");

  const stats = {
    total: emails.length,
    high: emails.filter((e) => e.urgency === "high").length,
    complaints: emails.filter((e) => e.category === "complaint").length,
    spam: emails.filter((e) => e.category === "spam").length,
  };

  const donutData = [
    { label: "Complaint", count: emails.filter((e) => e.category === "complaint").length, color: "#e24b4a" },
    { label: "Request", count: emails.filter((e) => e.category === "request").length, color: "#378add" },
    { label: "Feedback", count: emails.filter((e) => e.category === "feedback").length, color: "#639922" },
    { label: "Spam", count: emails.filter((e) => e.category === "spam").length, color: "#888780" },
  ];

  const filtered = emails.filter((e) => {
    if (filterCat !== "all" && e.category !== filterCat) return false;
    if (filterUrg !== "all" && e.urgency !== filterUrg) return false;
    if (search && !e.subject.toLowerCase().includes(search.toLowerCase()) && !e.body.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  async function classifyEmail() {
    if (!subject.trim() || !body.trim()) {
      setError("Please fill in both subject and body.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    setStreamText("Analyzing email...");

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [
            {
              role: "user",
              content: `Classify this customer support email. Return ONLY valid JSON with no markdown, no explanation.

Subject: ${subject}
Body: ${body}

Return JSON exactly like this:
{
  "category": "complaint|request|feedback|spam",
  "urgency": "high|medium|low",
  "confidence": 0.0-1.0,
  "sentiment": "positive|negative|neutral",
  "summary": "one sentence summary",
  "tags": ["tag1","tag2","tag3"]
}`,
            },
          ],
        }),
      });

      const data = await response.json();
      const text = data.content?.[0]?.text || "";
      const clean = text.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);

      setResult(parsed);
      setStreamText(parsed.summary);

      const newEmail = {
        id: Date.now(),
        subject,
        body,
        ...parsed,
        timestamp: new Date(),
      };
      setEmails((prev) => [newEmail, ...prev]);
      setNewId(newEmail.id);
      setSubject("");
      setBody("");
      setTimeout(() => setNewId(null), 5000);
    } catch (err) {
      setError("Classification failed. Please try again.");
      setStreamText("");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, sans-serif", background: "#fafaf8", minHeight: "100vh" }}>
      <style>{`
        @keyframes slideIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        * { box-sizing: border-box; }
        input, textarea, select { outline: none; font-family: inherit; }
        input:focus, textarea:focus { border-color: #378add !important; }
      `}</style>

      {/* Header */}
      <div style={{ background: "white", borderBottom: "0.5px solid #d3d1c7", padding: "0 24px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: 56 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 28, height: 28, background: "#2c2c2a", borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ color: "white", fontSize: 13 }}>✉</span>
            </div>
            <span style={{ fontSize: 14, fontWeight: 500, color: "#2c2c2a" }}>MailSense</span>
            <span style={{ fontSize: 11, color: "#b4b2a9", marginLeft: 4 }}>AI Classification</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#639922", animation: "pulse 2s infinite", display: "inline-block" }} />
            <span style={{ fontSize: 11, color: "#5f5e5a" }}>Claude AI active</span>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 24px" }}>
        {/* Stat cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
          {[
            { label: "Total Emails", value: stats.total, accent: "#2c2c2a" },
            { label: "High Priority", value: stats.high, accent: "#e24b4a" },
            { label: "Complaints", value: stats.complaints, accent: "#ba7517" },
            { label: "Spam Caught", value: stats.spam, accent: "#888780" },
          ].map((s) => (
            <div
              key={s.label}
              style={{ background: "white", border: "0.5px solid #d3d1c7", borderRadius: 10, padding: "14px 16px" }}
            >
              <div style={{ fontSize: 11, color: "#888780", marginBottom: 6 }}>{s.label}</div>
              <div style={{ fontSize: 28, fontWeight: 500, color: s.accent }}>{s.value}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20 }}>
          {/* Left column */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Search & Filters */}
            <div style={{ background: "white", border: "0.5px solid #d3d1c7", borderRadius: 10, padding: "14px 16px" }}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search emails..."
                  style={{
                    flex: 1,
                    minWidth: 160,
                    border: "0.5px solid #d3d1c7",
                    borderRadius: 7,
                    padding: "7px 12px",
                    fontSize: 13,
                    color: "#2c2c2a",
                  }}
                />
                <select
                  value={filterCat}
                  onChange={(e) => setFilterCat(e.target.value)}
                  style={{ border: "0.5px solid #d3d1c7", borderRadius: 7, padding: "7px 10px", fontSize: 12, color: "#5f5e5a", background: "white" }}
                >
                  <option value="all">All categories</option>
                  <option value="complaint">Complaint</option>
                  <option value="request">Request</option>
                  <option value="feedback">Feedback</option>
                  <option value="spam">Spam</option>
                </select>
                <select
                  value={filterUrg}
                  onChange={(e) => setFilterUrg(e.target.value)}
                  style={{ border: "0.5px solid #d3d1c7", borderRadius: 7, padding: "7px 10px", fontSize: 12, color: "#5f5e5a", background: "white" }}
                >
                  <option value="all">All urgency</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
            </div>

            {/* Email list */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {filtered.length === 0 ? (
                <div style={{ textAlign: "center", padding: "40px 0", color: "#b4b2a9", fontSize: 13 }}>
                  No emails match your filters.
                </div>
              ) : (
                filtered.map((email) => (
                  <EmailCard key={email.id} email={email} isNew={email.id === newId} />
                ))
              )}
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Classify panel */}
            <div style={{ background: "white", border: "0.5px solid #d3d1c7", borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "#2c2c2a", marginBottom: 12 }}>
                Classify new email
              </div>
              <input
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="Subject line"
                style={{
                  width: "100%",
                  border: "0.5px solid #d3d1c7",
                  borderRadius: 7,
                  padding: "8px 12px",
                  fontSize: 12,
                  color: "#2c2c2a",
                  marginBottom: 8,
                }}
              />
              <textarea
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Email body..."
                rows={5}
                style={{
                  width: "100%",
                  border: "0.5px solid #d3d1c7",
                  borderRadius: 7,
                  padding: "8px 12px",
                  fontSize: 12,
                  color: "#2c2c2a",
                  resize: "vertical",
                  marginBottom: 10,
                  lineHeight: 1.5,
                }}
              />
              {error && <p style={{ fontSize: 11, color: "#e24b4a", marginBottom: 8 }}>{error}</p>}
              <button
                onClick={classifyEmail}
                disabled={loading}
                style={{
                  width: "100%",
                  background: loading ? "#d3d1c7" : "#2c2c2a",
                  color: "white",
                  border: "none",
                  borderRadius: 7,
                  padding: "9px 0",
                  fontSize: 12,
                  fontWeight: 500,
                  cursor: loading ? "not-allowed" : "pointer",
                  transition: "background 0.15s",
                }}
              >
                {loading ? "Analyzing..." : "Classify with Claude AI"}
              </button>

              {loading && (
                <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ display: "flex", gap: 3 }}>
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        style={{
                          width: 5,
                          height: 5,
                          borderRadius: "50%",
                          background: "#378add",
                          animation: `pulse 1.2s ${i * 0.2}s infinite`,
                          display: "inline-block",
                        }}
                      />
                    ))}
                  </div>
                  <span style={{ fontSize: 11, color: "#888780" }}>Reading email content...</span>
                </div>
              )}

              {result && !loading && (
                <div style={{ marginTop: 12, background: "#eaf3de", borderRadius: 8, padding: 12 }}>
                  <div style={{ fontSize: 11, color: "#3b6d11", fontWeight: 500, marginBottom: 8 }}>
                    ✓ Classification complete
                  </div>
                  <div style={{ fontSize: 12, color: "#27500a", lineHeight: 1.6 }}>
                    <TypewriterText text={result.summary} />
                  </div>
                  <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
                    <CategoryPill category={result.category} />
                    <UrgencyDot urgency={result.urgency} />
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <MiniBar value={result.confidence} color="#3b6d11" />
                  </div>
                </div>
              )}
            </div>

            {/* Distribution chart */}
            <div style={{ background: "white", border: "0.5px solid #d3d1c7", borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "#2c2c2a", marginBottom: 14 }}>
                Category breakdown
              </div>
              <DonutChart data={donutData} />
            </div>

            {/* Urgency bars */}
            <div style={{ background: "white", border: "0.5px solid #d3d1c7", borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "#2c2c2a", marginBottom: 14 }}>
                Urgency distribution
              </div>
              {["high", "medium", "low"].map((u) => {
                const count = emails.filter((e) => e.urgency === u).length;
                const cfg = URGENCY_CONFIG[u];
                return (
                  <div key={u} style={{ marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 11, color: "#5f5e5a" }}>{cfg.label}</span>
                      <span style={{ fontSize: 11, color: "#888780" }}>{count}</span>
                    </div>
                    <MiniBar value={count / (emails.length || 1)} color={cfg.dot} />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
