export default function SentimentBadge({ sentiment }) {
  const map = {
    Positive: { bg: "#e7f8ee", color: "#137a41" },
    Neutral:  { bg: "#f2f4f7", color: "#475467" },
    Negative: { bg: "#fdecec", color: "#b42318" },
  };
  const s = map[sentiment] || map.Neutral;
  return (
    <span style={{
      background: s.bg, color: s.color, padding: "6px 10px",
      borderRadius: 999, fontSize: 14, fontWeight: 600
    }}>
      {sentiment}
    </span>
  );
}
