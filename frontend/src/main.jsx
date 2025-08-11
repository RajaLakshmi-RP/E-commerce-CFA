import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import './styles.css';

// Force-refresh once if a newer index.html is available on the server
async function ensureFreshIndex() {
  try {
    const LOCAL_ID =
      document.querySelector('meta[name="build-id"]')?.content || '';
    const RELOAD_FLAG = '__refreshed_for_new_build__';

    // Fetch latest index.html bypassing cache
    const res = await fetch('index.html', { cache: 'no-store' });
    if (!res.ok) return;
    const text = await res.text();
    const match = text.match(/<meta name="build-id" content="([^"]+)">/i);
    const REMOTE_ID = match ? match[1] : '';

    // If remote is newer and we haven't refreshed yet, reload once
    if (REMOTE_ID && REMOTE_ID !== LOCAL_ID && !sessionStorage.getItem(RELOAD_FLAG)) {
      sessionStorage.setItem(RELOAD_FLAG, '1');
      location.replace(`index.html?nocache=${Date.now()}`);
      return;
    }
  } catch {
    // ignore network errors; app will still boot
  }
}

async function boot() {
  await ensureFreshIndex();
  ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

boot();
