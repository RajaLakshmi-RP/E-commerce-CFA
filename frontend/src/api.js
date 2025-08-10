const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8080";

export async function predictReview(reviewText) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_text: reviewText }),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

export async function rephraseReview({ reviewText, tone = "neutral", length = "medium", sentiment }) {
  const res = await fetch(`${API_BASE}/rephrase`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_text: reviewText, tone, length, sentiment }),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}
