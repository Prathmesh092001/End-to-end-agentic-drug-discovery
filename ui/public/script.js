const $ = (id) => document.getElementById(id);

const state = {
  apiEndpoint: localStorage.getItem("partex_api_endpoint") || "http://localhost:8000",
  history: JSON.parse(localStorage.getItem("partex_research_history") || "[]"),
  running: false,
  trace: []
};

const agents = [
  ["supervisor", "Supervisor", "Planning & coordination"],
  ["retrieval", "Retrieval", "Search & gather"],
  ["profiling", "Profiling", "Target & compound"],
  ["risk", "Risk scoring", "ADMET & safety"],
  ["hypothesis", "Hypothesis", "Generate insights"],
  ["competitive", "Competitive intel", "Market intelligence"],
  ["critic", "Critic", "Validate & refine"]
];

const templates = {
  target: "Analyze the target biology, protein function, disease relevance, pathways, mutations, structural context, known biological evidence, and therapeutic opportunities. Ground the findings in scientific sources.",
  compound: "Analyze this compound including mechanism of action, target interactions, known potency/activity evidence, chemical context, ADMET considerations, safety signals, and development context. Cite supporting evidence.",
  resistance: "Investigate mechanisms of therapeutic resistance associated with this target or compound, including mutations, structural mechanisms, known resistance patterns, alternative strategies, and supporting literature.",
  competitive: "Analyze the competitive landscape for this target or therapeutic area, including known compounds, mechanisms, development signals, competitive programs, evidence strength, and open opportunities.",
  literature: "Conduct an evidence-grounded literature review covering biological context, therapeutic hypotheses, key findings, conflicting evidence, research gaps, and relevant compounds or targets."
};

function setApiStatus(text, healthy = true) {
  $("apiStatus").textContent = text;
  $("railApi").textContent = text;
  $("copilotStatus").textContent = healthy ? "Online" : "Offline";
}

async function checkHealth() {
  try {
    const r = await fetch(`${state.apiEndpoint.replace(/\/$/, "")}/health`, { cache: "no-store" });
    if (!r.ok) throw new Error("unhealthy");
    setApiStatus("Healthy", true);
  } catch {
    setApiStatus("Offline", false);
  }
}

function updateCount() {
  $("charCount").textContent = `${$("question").value.length} chars`;
}

function showToast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => el.classList.remove("show"), 2600);
}

function setAgent(agentId, status) {
  const card = document.querySelector(`[data-agent="${agentId}"]`);
  if (!card) return;
  card.classList.remove("active", "running", "done");
  const label = card.querySelector(".agent-status");

  if (status === "running") {
    card.classList.add("running");
    label.textContent = "Running";
  } else if (status === "done") {
    card.classList.add("done");
    label.textContent = "Completed";
  } else {
    label.textContent = "Waiting";
  }
}

function resetAgents() {
  agents.forEach(([id], i) => setAgent(id, i === 0 ? "active" : "waiting"));
  $("executionState").innerHTML = '<span class="state-dot"></span> Ready';
}

function addTrace(agent, description, time = new Date()) {
  state.trace.push({
    time: time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    agent,
    description
  });
  renderTrace();
}

function renderTrace() {
  const list = $("traceList");
  if (!state.trace.length) {
    list.innerHTML = '<div class="trace-empty">No research run yet.</div>';
    return;
  }
  list.innerHTML = state.trace.map(item => `
    <div class="trace-item">
      <div class="trace-time">${escapeHtml(item.time)}</div>
      <div class="trace-marker"></div>
      <div class="trace-body">
        <b>${escapeHtml(item.agent)}</b>
        <small>${escapeHtml(item.description)}</small>
      </div>
    </div>
  `).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[ch]));
}

function extractInsights(report, target) {
  const clean = String(report || "").replace(/[#*_`]/g, " ").replace(/\s+/g, " ").trim();
  const sentences = clean.split(/(?<=[.!?])\s+/).filter(s => s.length > 45);
  const preferred = sentences.filter(s =>
    /(EGFR|KRAS|target|compound|inhibitor|potency|IC50|ADMET|risk|resistance|evidence|competitive|mechanism|mutation)/i.test(s)
  );
  const picks = [...preferred, ...sentences].filter((v, i, a) => a.indexOf(v) === i).slice(0, 5);

  const list = $("insightsList");
  if (!picks.length) {
    list.innerHTML = `<li>Research completed for ${escapeHtml(target)}. Review the synthesis and evidence tabs for details.</li>`;
    return;
  }
  list.innerHTML = picks.map(s => `<li>${escapeHtml(s.slice(0, 260))}${s.length > 260 ? "…" : ""}</li>`).join("");
}

function renderReport(report) {
  const raw = report || "No report returned.";
  if (window.marked) {
    $("report").innerHTML = marked.parse(raw);
  } else {
    $("report").innerHTML = `<pre>${escapeHtml(raw)}</pre>`;
  }
}

function saveHistory(target, question, result) {
  const entry = {
    id: Date.now(),
    target,
    question,
    approved: !!result.approved,
    score: Number(result.groundedness_score || 0),
    time: new Date().toLocaleString()
  };
  state.history.unshift(entry);
  state.history = state.history.slice(0, 8);
  localStorage.setItem("partex_research_history", JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const list = $("historyList");
  const rail = $("railHistory");

  if (!state.history.length) {
    list.innerHTML = '<div class="history-empty">Your recent research runs will appear here in this browser.</div>';
    return;
  }

  list.innerHTML = state.history.map(h => `
    <div class="history-entry" data-history-id="${h.id}">
      <div><b>${escapeHtml(h.target)} research</b><small>${escapeHtml(h.question.slice(0, 150))} · ${escapeHtml(h.time)}</small></div>
      <span>${h.approved ? "Approved" : "Review"} · ${h.score.toFixed(2)}</span>
    </div>
  `).join("");

  rail.innerHTML = state.history.slice(0, 4).map((h, i) => `
    <div class="rail-item ${i === 0 ? "selected" : ""}">
      <span class="rail-line"></span>
      <div><b>${escapeHtml(h.target)} research</b><small>${escapeHtml(h.time)}</small></div>
      <em>${h.approved ? "Approved" : "Review"}</em>
    </div>
  `).join("");

  document.querySelectorAll("[data-history-id]").forEach(el => {
    el.addEventListener("click", () => {
      const item = state.history.find(h => String(h.id) === el.dataset.historyId);
      if (!item) return;
      $("compound").value = item.target;
      $("question").value = item.question;
      updateCount();
      $("research").scrollIntoView({ behavior: "smooth", block: "start" });
      showToast("Previous research loaded into the workspace.");
    });
  });
}

function activateTab(tabId) {
  document.querySelectorAll(".result-tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.toggle("active", panel.id === tabId));
}

function simulateAgentProgress() {
  const timings = [0, 750, 1600, 2600, 3700, 5000, 6300];
  state.trace = [];
  renderTrace();

  timings.forEach((delay, index) => {
    setTimeout(() => {
      if (!state.running) return;
      const [id, name, description] = agents[index];
      if (index > 0) setAgent(agents[index - 1][0], "done");
      setAgent(id, "running");
      $("executionState").innerHTML = `<span class="state-dot"></span> ${name} running`;
      addTrace(name, `${description}. Processing current research request.`);
    }, delay);
  });
}

async function runResearch() {
  if (state.running) return;

  const target = $("compound").value.trim();
  const question = $("question").value.trim();

  if (!target) {
    showToast("Enter a target or compound first.");
    $("compound").focus();
    return;
  }
  if (!question) {
    showToast("Enter a scientific research question.");
    $("question").focus();
    return;
  }

  state.running = true;
  $("runBtn").disabled = true;
  $("resultState").textContent = "Running";
  $("approval").textContent = "In review";
  $("groundedness").textContent = "—";
  $("visualTarget").textContent = target;
  $("biologyTarget").textContent = target;
  $("report").innerHTML = '<div class="empty-state">Agents are executing the research pipeline…</div>';
  $("insightsList").innerHTML = "<li>Retrieving scientific evidence…</li><li>Profiling biological context…</li><li>Evaluating risk and opportunities…</li>";
  resetAgents();
  $("executionState").innerHTML = '<span class="state-dot"></span> Starting';
  simulateAgentProgress();

  const endpoint = `${state.apiEndpoint.replace(/\/$/, "")}/v1/drug-intelligence`;
  const started = Date.now();

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, compound_or_target: target })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Server responded with ${response.status}`);
    }

    const data = await response.json();

    state.running = false;
    agents.forEach(([id]) => setAgent(id, "done"));
    $("executionState").innerHTML = '<span class="state-dot"></span> Completed';
    $("resultState").textContent = "Completed";

    const score = Number(data.groundedness_score || 0);
    $("groundedness").textContent = Number.isFinite(score) ? score.toFixed(2) : "N/A";
    $("approval").textContent = data.approved ? "✓ Approved" : "Review required";

    renderReport(data.final_report);
    extractInsights(data.final_report, target);

    addTrace("Final report", data.approved
      ? `Critic gate approved the response. Groundedness score: ${score.toFixed(2)}.`
      : `Critic gate returned a review-needed result. Groundedness score: ${score.toFixed(2)}.`);

    saveHistory(target, question, data);
    setApiStatus("Healthy", true);
    showToast(`Research completed in ${((Date.now() - started) / 1000).toFixed(1)}s.`);
    activateTab("resultsTab");
  } catch (err) {
    state.running = false;
    $("runBtn").disabled = false;
    $("executionState").innerHTML = '<span class="state-dot" style="background:var(--red)"></span> Failed';
    $("resultState").textContent = "Failed";
    $("approval").textContent = "Unavailable";
    $("report").innerHTML = `<div class="empty-state" style="color:var(--red)">Connection error: ${escapeHtml(err.message)}<br><br>Ensure FastAPI is running on port 8000.</div>`;
    addTrace("Request error", err.message);
    setApiStatus("Offline", false);
    showToast("Research request failed. Check the API.");
  } finally {
    $("runBtn").disabled = false;
  }
}

$("question").addEventListener("input", updateCount);
$("runBtn").addEventListener("click", runResearch);

document.querySelectorAll(".chip[data-target]").forEach(chip => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach(c => c.classList.remove("active-chip"));
    chip.classList.add("active-chip");
    $("compound").value = chip.dataset.target;
    $("visualTarget").textContent = chip.dataset.target;
    $("biologyTarget").textContent = chip.dataset.target;
  });
});

$("addEntity").addEventListener("click", () => {
  const entity = prompt("Add a research entity, e.g. T790M or osimertinib:");
  if (!entity) return;
  const chip = document.createElement("button");
  chip.className = "chip active-chip";
  chip.textContent = entity;
  chip.addEventListener("click", () => {
    $("compound").value = entity;
    $("visualTarget").textContent = entity;
    $("biologyTarget").textContent = entity;
  });
  document.querySelector(".chips").insertBefore(chip, $("addEntity"));
  showToast(`${entity} added to the research context.`);
});

document.querySelectorAll(".template").forEach(btn => {
  btn.addEventListener("click", () => {
    $("question").value = templates[btn.dataset.template];
    updateCount();
    $("question").focus();
    showToast(`${btn.textContent} template loaded.`);
  });
});

document.querySelectorAll(".result-tab").forEach(btn => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});

document.querySelectorAll(".nav-item[data-scroll]").forEach(btn => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.scroll);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    btn.classList.add("active");
  });
});

$("viewHistory").addEventListener("click", () => $("history").scrollIntoView({ behavior: "smooth" }));
$("clearHistory").addEventListener("click", () => {
  state.history = [];
  localStorage.removeItem("partex_research_history");
  renderHistory();
  showToast("Local research history cleared.");
});

$("settingsBtn").addEventListener("click", () => {
  $("apiEndpoint").value = state.apiEndpoint;
  $("settingsModal").classList.add("open");
  $("settingsModal").setAttribute("aria-hidden", "false");
});
$("closeSettings").addEventListener("click", () => {
  $("settingsModal").classList.remove("open");
  $("settingsModal").setAttribute("aria-hidden", "true");
});
$("saveSettings").addEventListener("click", () => {
  const value = $("apiEndpoint").value.trim().replace(/\/$/, "");
  if (!value) return;
  state.apiEndpoint = value;
  localStorage.setItem("partex_api_endpoint", value);
  $("settingsModal").classList.remove("open");
  showToast("API endpoint saved.");
  checkHealth();
});

updateCount();
renderHistory();
resetAgents();
checkHealth();
