// src/components/SentimentBadge.jsx
import React from "react";

export function SentimentBadge({ sentiment }) {
  const label = sentiment ?? "unknown";

  // Neon color palette
  const color =
    label.toLowerCase() === "positive"
      ? "#39FF14" // neon green
      : label.toLowerCase() === "negative"
      ? "#FF10F0" // neon pink
      : "#FFFF33"; // neon yellow

  const pillStyle = {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    padding: "6px 12px",
    borderRadius: "9999px",
    border: `1px solid ${color}`,
    background: "#0d0d0d", // dark background for neon contrast
    fontSize: 14,
    fontWeight: 600,
    textTransform: "capitalize",
    color: color, // text in neon color
    boxShadow: `0 0 8px ${color}, 0 0 16px ${color}`, // glow effect
  };

  const dotStyle = {
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: color,
    display: "inline-block",
    boxShadow: `0 0 4px ${color}, 0 0 8px ${color}`,
  };

  return (
    <span style={pillStyle} title={`Sentiment: ${label}`}>
      <span style={dotStyle} />
      {label}
    </span>
  );
}
