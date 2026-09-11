import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const app = express();
const PORT = process.env.PORT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Proxy endpoint: the browser only ever talks to this same-origin route.
// This server forwards the request to the FastAPI backend server-to-server,
// which sidesteps browser CORS restrictions entirely - no changes needed
// to app.py, and the FastAPI URL never has to be exposed to client-side code.
app.post("/api/drug-intelligence", async (req, res) => {
  try {
    const response = await fetch(`${FASTAPI_URL}/v1/drug-intelligence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (err) {
    console.error("Error contacting FastAPI backend:", err.message);
    res.status(502).json({
      detail: "Could not reach the FastAPI backend. Is `python app.py` running on port 8000?",
    });
  }
});

app.listen(PORT, () => {
  console.log(`UI server running at http://localhost:${PORT}`);
  console.log(`Proxying agent requests to ${FASTAPI_URL}`);
});