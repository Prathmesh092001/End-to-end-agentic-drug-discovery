const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const reportEl = document.getElementById("report");
const compoundInput = document.getElementById("compound");
const questionInput = document.getElementById("question");

runBtn.addEventListener("click", async () => {
  const compound = compoundInput.value.trim();
  const question = questionInput.value.trim();

  if (reportEl) {
    reportEl.style.display = "none";
    reportEl.innerHTML = "";
  }

  if (!compound) {
    statusEl.innerHTML = `<span class="error-text">Please enter a compound or target name.</span>`;
    return;
  }

  runBtn.disabled = true;
  statusEl.innerHTML = `<span class="spinner"></span>Running the 7-agent pipeline - this typically takes 30-90 seconds...`;

  try {
    // Connect directly to local FastAPI instance
    const response = await fetch("http://localhost:8000/v1/drug-intelligence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, compound_or_target: compound }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server responded with status ${response.status}`);
    }

    const data = await response.json();

    const groundednessScore = data.groundedness_score !== undefined ? data.groundedness_score.toFixed(2) : "N/A";
    const badgeClass = data.approved ? "ok" : "warn";
    const badgeText = data.approved
      ? `Approved by the critic agent - groundedness score ${groundednessScore}`
      : `Did NOT pass the groundedness check (score ${groundednessScore}) - read with caution`;

    statusEl.innerHTML = `<span class="badge ${badgeClass}">${badgeText}</span>`;

    const rawReport = data.final_report || "*No report returned.*";
    if (reportEl) {
      reportEl.innerHTML = typeof marked !== "undefined" ? marked.parse(rawReport) : `<pre>${rawReport}</pre>`;
      reportEl.style.display = "block";
    }
  } catch (err) {
    statusEl.innerHTML = `<span class="error-text">Connection Error: ${err.message}. Ensure FastAPI is running on port 8000.</span>`;
  } finally {
    runBtn.disabled = false;
  }
});