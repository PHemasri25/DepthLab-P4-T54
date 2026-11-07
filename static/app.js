"use strict";

const state = { rgb: null, depth: null, truth: null, results: [] };
const byId = (id) => document.getElementById(id);
const methodNames = {
  nearest: "Nearest neighbour",
  bilinear: "Bilinear",
  bicubic: "Bicubic",
  joint_bilateral: "Joint bilateral",
  guided: "Guided filter",
  cnn: "Residual U-Net",
  hybrid: "Hybrid (Bicubic + guided + U-Net)",
};

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function showAlert(message) {
  const alert = byId("workbench-alert");
  alert.textContent = message;
  alert.classList.remove("hidden");
}

function clearAlert() {
  byId("workbench-alert").classList.add("hidden");
  byId("workbench-alert").textContent = "";
}

function setInput(name, file) {
  state[name] = file;
  const id = name === "rgb" ? "rgb-input" : name === "depth" ? "depth-input" : "truth-input";
  const input = byId(id);
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  const preview = byId(`${name}-preview`);
  const filename = byId(`${name}-name`);
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  filename.textContent = file.name;
  filename.title = file.name;
  updateInputPreviews();
}

function updateInputPreviews() {
  const anyInput = state.rgb || state.depth || state.truth;
  byId("input-previews").classList.toggle("hidden", !anyInput);
  if (state.rgb) byId("preview-rgb").src = URL.createObjectURL(state.rgb);
  if (state.depth) byId("preview-sparse").src = URL.createObjectURL(state.depth);
  byId("preview-truth").classList.toggle("hidden", !state.truth);
  byId("truth-missing").classList.toggle("hidden", Boolean(state.truth));
  if (state.truth) byId("preview-truth").src = URL.createObjectURL(state.truth);
}

function attachUpload(name) {
  const inputId = name === "rgb" ? "rgb-input" : name === "depth" ? "depth-input" : "truth-input";
  const input = byId(inputId);
  const card = input.closest(".upload-card");
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) setInput(name, input.files[0]);
  });
  card.addEventListener("dragover", (event) => {
    event.preventDefault();
    card.classList.add("dragover");
  });
  card.addEventListener("dragleave", () => card.classList.remove("dragover"));
  card.addEventListener("drop", (event) => {
    event.preventDefault();
    card.classList.remove("dragover");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) setInput(name, file);
    else showAlert("Choose an image file in PNG, JPEG or WebP format.");
  });
}

function decodeBase64File(data, filename, mimeType = "image/png") {
  const raw = atob(data);
  const bytes = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) bytes[index] = raw.charCodeAt(index);
  return new File([bytes], filename, { type: mimeType });
}

async function loadDemo() {
  const button = byId("demo-button");
  button.disabled = true;
  clearAlert();
  try {
    const response = await fetch("/api/demo");
    if (!response.ok) throw new Error(`Sample scene request failed (${response.status}).`);
    const sample = await response.json();
    setInput("rgb", decodeBase64File(sample.rgb, "generated-rgb.png"));
    setInput("depth", decodeBase64File(sample.sparse_depth, "generated-sparse-depth.png"));
    setInput("truth", decodeBase64File(sample.ground_truth, "generated-ground-truth.png"));
    byId("scale-select").value = String(sample.scale);
    byId("method-select").value = "compare";
    await runReconstruction();
  } catch (error) {
    showAlert(error.message || "Could not load the sample scene.");
  } finally {
    button.disabled = false;
  }
}

function metricMarkup(result, unit) {
  const metrics = result.metrics || {};
  const cells = result.metrics_available
    ? [
        ["RMSE", `${formatNumber(metrics.rmse)} ${unit}`],
        ["MAE", `${formatNumber(metrics.mae)} ${unit}`],
        ["PSNR", `${formatNumber(metrics.psnr, 2)} dB`],
        ["SSIM", formatNumber(metrics.ssim, 3)],
        ["EDGE", `${formatNumber(metrics.edge_preservation * 100, 1)}%`],
        ["BOUNDARY", `${formatNumber(metrics.boundary_accuracy * 100, 1)}%`],
        ["EDGE F1", `${formatNumber(metrics.edge_f1 * 100, 1)}%`],
      ]
    : [["RMSE / MAE", "Ground truth needed"], ["EDGE / RGB", `${formatNumber(metrics.edge_preservation * 100, 1)}%`]];
  return `<div class="mini-metrics">${cells.map(([label, value]) => `<span>${label}<b>${value}</b></span>`).join("")}<span>LATENCY<b>${formatNumber(result.latency_ms)} ms</b></span><span>PROCESS RSS<b>${formatNumber(result.process_rss_mb)} MB</b></span></div>`;
}

function renderResults(payload) {
  state.results = payload.results;
  byId("results-heading").classList.remove("hidden");
  byId("results-title").textContent = payload.results.length > 1 ? "Seven-method comparison" : payload.results[0].label;
  byId("result-meta").textContent = `${payload.input.width} × ${payload.input.height} output · ${payload.input.scale}× target`;
  byId("results-grid").innerHTML = payload.results.map((result, index) => {
    const errorFigure = result.error_png
      ? `<figure><figcaption>ABSOLUTE ERROR</figcaption><img src="data:image/png;base64,${result.error_png}" alt="${result.label} error map"></figure>`
      : "";
    return `<article class="result-card">
      <h4><span>${result.label}</span><span class="method-index">0${index + 1}</span></h4>
      <div class="result-images">
        <figure><figcaption>REFINED DEPTH</figcaption><img src="data:image/png;base64,${result.color_png}" alt="${result.label} depth map"></figure>
        <figure><figcaption>RGB / DEPTH / ALIGNED EDGES</figcaption><img src="data:image/png;base64,${result.edge_png}" alt="${result.label} edge visualization"></figure>
        ${errorFigure}
      </div>
      ${metricMarkup(result, payload.unit)}
      <a class="download-link" href="data:image/png;base64,${result.depth_png}" download="depthlab-${result.method}-depth-16bit.png">↓ &nbsp; Download 16-bit depth PNG</a>
    </article>`;
  }).join("");
  renderHistory(payload);
  byId("results-heading").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runReconstruction() {
  clearAlert();
  if (!state.rgb || !state.depth) {
    showAlert("Upload both an RGB guidance image and a sparse depth image, or load the sample scene.");
    return;
  }
  const button = byId("run-button");
  button.disabled = true;
  button.innerHTML = '<span class="run-icon">◌</span> Processing…';
  byId("workbench-loading").classList.remove("hidden");
  try {
    const form = new FormData();
    form.append("rgb_file", state.rgb);
    form.append("depth_file", state.depth);
    if (state.truth) form.append("ground_truth_file", state.truth);
    form.append("method", byId("method-select").value);
    form.append("scale", byId("scale-select").value);
    form.append("radius", byId("radius-input").value);
    form.append("epsilon", "0.01");
    const response = await fetch("/api/process", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Reconstruction failed (${response.status}).`);
    renderResults(payload);
  } catch (error) {
    showAlert(error.message || "Reconstruction failed. Check the images and try again.");
  } finally {
    byId("workbench-loading").classList.add("hidden");
    button.disabled = false;
    button.innerHTML = '<span class="run-icon">▶</span> Reconstruct';
  }
}

function renderHistory(payload) {
  let history = [];
  try {
    const saved = JSON.parse(localStorage.getItem("depthlab-history") || "[]");
    history = Array.isArray(saved) ? saved.filter((item) => item && typeof item === "object") : [];
  } catch (error) {
    console.warn("Could not read local run history.", error);
  }
  const best = payload.results
    .filter((result) => result.metrics_available)
    .sort((left, right) => left.metrics.rmse - right.metrics.rmse)[0];
  const row = {
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    method: payload.results.length > 1 ? "7-method comparison" : payload.results[0].label,
    size: `${payload.input.width} × ${payload.input.height}`,
    rmse: best ? `${formatNumber(best.metrics.rmse)} mm` : "Ground truth needed",
    status: "Complete",
  };
  history = [row, ...history].slice(0, 6);
  try {
    localStorage.setItem("depthlab-history", JSON.stringify(history));
  } catch (error) {
    console.warn("Could not save local run history.", error);
  }
  drawHistory(history);
}

function drawHistory(history) {
  const body = byId("history-table");
  if (!history.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty-row">Runs you complete will appear here.</td></tr>';
    return;
  }
  body.replaceChildren();
  for (const row of history) {
    const tableRow = document.createElement("tr");
    for (const [value, className] of [
      [row.time, ""],
      [row.method, "table-method"],
      [row.size, ""],
      [row.rmse, "number-cell"],
    ]) {
      const cell = document.createElement("td");
      cell.textContent = typeof value === "string" ? value : "";
      if (className) cell.className = className;
      tableRow.append(cell);
    }
    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = "rank-pill";
    status.textContent = typeof row.status === "string" ? row.status : "";
    statusCell.append(status);
    tableRow.append(statusCell);
    body.append(tableRow);
  }
}

function drawChart(rows) {
  const canvas = byId("rmse-chart");
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !rows.length) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(bounds.width * ratio);
  canvas.height = Math.floor(bounds.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = bounds.width;
  const height = bounds.height;
  const left = 112;
  const right = 42;
  const top = 7;
  const rowHeight = (height - top - 4) / rows.length;
  const max = Math.max(...rows.map((row) => row.rmse + (row.rmse_std || 0))) * 1.08 || 1;
  const plotWidth = width - left - right;
  context.font = '9px "DM Mono", monospace';
  rows.forEach((row, index) => {
    const y = top + index * rowHeight + rowHeight * 0.5;
    const valueWidth = (row.rmse / max) * plotWidth;
    context.fillStyle = "#8b9caf";
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(row.label, left - 11, y);
    context.fillStyle = row.method === "hybrid" ? "#69e0c3" : "#438e8d";
    context.beginPath();
    context.roundRect(left, y - 5, Math.max(valueWidth, 2), 10, 3);
    context.fill();
    const errorWidth = ((row.rmse_std || 0) / max) * plotWidth;
    context.strokeStyle = "#b8c5d2";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left + Math.max(0, valueWidth - errorWidth), y);
    context.lineTo(left + Math.min(plotWidth, valueWidth + errorWidth), y);
    context.stroke();
    context.fillStyle = "#b5c3d0";
    context.textAlign = "left";
    context.fillText(`${formatNumber(row.rmse, 0)} ± ${formatNumber(row.rmse_std, 0)}`, left + valueWidth + 7, y);
  });
}

function renderBenchmark(report) {
  const rows = [...report.results].sort((a, b) => a.rmse - b.rmse);
  renderScaleBenchmark(report.results_by_scale);
  const best = rows[0];
  const meanLatency = rows.reduce((sum, row) => sum + row.latency_ms, 0) / rows.length;
  byId("stat-rmse").textContent = `${formatNumber(best.rmse)} mm`;
  byId("stat-rmse-method").textContent = `Best: ${best.label}`;
  byId("stat-edge").textContent = `${formatNumber(best.edge_preservation * 100, 1)}%`;
  byId("stat-scenes").textContent = String(report.scenes);
  byId("stat-split").textContent = "Held-out scenes · no overlap";
  byId("stat-latency").textContent = `${formatNumber(meanLatency)} ms`;
  byId("split-note").textContent = report.split;
  byId("benchmark-scale").textContent = `${report.scale}× upsampling · ${report.unit}`;
  byId("benchmark-limitations").textContent = report.limitations;

  const body = byId("benchmark-table");
  const meanStd = (mean, std, digits = 1) => `${formatNumber(mean, digits)} ± ${formatNumber(std, digits)}`;
  body.innerHTML = rows.map((row, index) => `<tr>
    <td class="table-method">${row.label} ${index === 0 ? '<span class="rank-pill">BEST RMSE</span>' : ""}</td>
    <td class="number-cell ${index === 0 ? "best-cell" : ""}">${meanStd(row.rmse, row.rmse_std)}</td>
    <td class="number-cell">${meanStd(row.mae, row.mae_std)}</td>
    <td class="number-cell">${meanStd(row.psnr, row.psnr_std, 2)}</td>
    <td class="number-cell">${meanStd(row.ssim, row.ssim_std, 3)}</td>
    <td class="number-cell">${meanStd(row.edge_preservation * 100, row.edge_preservation_std * 100)}%</td>
    <td class="number-cell">${meanStd(row.boundary_accuracy * 100, row.boundary_accuracy_std * 100)}%</td>
    <td class="number-cell">${meanStd(row.edge_f1 * 100, row.edge_f1_std * 100)}%</td>
    <td class="number-cell">${meanStd(row.latency_ms, row.latency_ms_std)} ms</td>
    <td class="number-cell">${meanStd(row.process_rss_mb, row.process_rss_mb_std)} MB</td>
  </tr>`).join("");
  renderExperiments(report.experiments);
  renderAblation(report.ablation, report.failure_analysis);
  drawChart(rows);
}

function renderAblation(rows, failureAnalysis) {
  const body = byId("ablation-table");
  body.innerHTML = (rows || []).map((row) => `<tr>
    <td class="table-method">${row.variant}</td>
    <td class="number-cell">${formatNumber(row.rmse)} ± ${formatNumber(row.rmse_std)}</td>
    <td class="number-cell">${formatNumber(row.mae)} ± ${formatNumber(row.mae_std)}</td>
    <td class="number-cell">${formatNumber(row.ssim, 3)} ± ${formatNumber(row.ssim_std, 3)}</td>
    <td class="number-cell">${formatNumber(row.edge_f1 * 100, 1)} ± ${formatNumber(row.edge_f1_std * 100, 1)}%</td>
  </tr>`).join("");
  const list = byId("failure-analysis");
  list.replaceChildren();
  for (const note of failureAnalysis || []) {
    const item = document.createElement("li");
    item.textContent = note;
    list.append(item);
  }
}

function renderExperiments(experiments) {
  const labels = {
    high_noise: "High noise (36 mm)",
    high_missing: "20% missing",
  };
  const rows = Object.entries(experiments?.results || {}).flatMap(([condition, methods]) =>
    methods.map((row) => `<tr><td>${labels[condition] || condition}</td><td class="table-method">${row.label}</td><td class="number-cell">${formatNumber(row.rmse)} ± ${formatNumber(row.rmse_std)}</td><td class="number-cell">${formatNumber(row.mae)} ± ${formatNumber(row.mae_std)}</td><td class="number-cell">${formatNumber(row.ssim, 3)} ± ${formatNumber(row.ssim_std, 3)}</td><td class="number-cell">${formatNumber(row.edge_f1 * 100, 1)} ± ${formatNumber(row.edge_f1_std * 100, 1)}%</td></tr>`),
  );
  byId("experiment-table").innerHTML = rows.join("") || '<tr><td colspan="6" class="empty-row">No robustness experiment results are available.</td></tr>';
}

function renderScaleBenchmark(resultsByScale) {
  const methods = resultsByScale?.["4"] || [];
  const scaleRows = [2, 4, 8].map((scale) => new Map(
    (resultsByScale?.[String(scale)] || []).map((row) => [row.method, row]),
  ));
  const body = byId("scale-benchmark-table");
  body.innerHTML = methods.map((method) => {
    const values = scaleRows.map((rows) => rows.get(method.method));
    if (values.some((row) => !row)) return "";
    return `<tr><td class="table-method">${method.label}</td>${values.map((row) => `<td class="number-cell">${formatNumber(row.rmse)}</td><td class="number-cell">${formatNumber(row.ssim, 3)}</td>`).join("")}</tr>`;
  }).join("");
}

async function loadBenchmark() {
  try {
    const response = await fetch("/api/benchmark");
    if (!response.ok) throw new Error(`Benchmark request failed (${response.status}).`);
    renderBenchmark(await response.json());
  } catch (error) {
    byId("stat-rmse-method").textContent = "Benchmark unavailable";
    byId("benchmark-table").innerHTML = `<tr><td colspan="10" class="empty-row">${error.message}</td></tr>`;
    byId("experiment-table").innerHTML = '<tr><td colspan="6" class="empty-row">Robustness results are unavailable.</td></tr>';
    byId("ablation-table").innerHTML = '<tr><td colspan="5" class="empty-row">Ablation results are unavailable.</td></tr>';
    byId("failure-analysis").innerHTML = "<li>Failure analysis is unavailable.</li>";
    byId("scale-benchmark-table").innerHTML = `<tr><td colspan="7" class="empty-row">The multi-scale benchmark is unavailable.</td></tr>`;
    byId("benchmark-limitations").textContent = "Could not load the computed benchmark. Check that the processing API is running.";
  }
}

async function checkApi() {
  const status = byId("api-status");
  const indicator = document.querySelector(".status-card .status-dot");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("unavailable");
    status.textContent = "Connected · 7 methods";
    indicator.classList.add("connected");
  } catch {
    status.textContent = "API unavailable";
  }
}

document.querySelectorAll(".side-nav a").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".side-nav a").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});
attachUpload("rgb");
attachUpload("depth");
attachUpload("truth");
byId("radius-input").addEventListener("input", (event) => {
  byId("radius-value").textContent = `${event.target.value} px`;
});
byId("demo-button").addEventListener("click", loadDemo);
byId("run-button").addEventListener("click", runReconstruction);
window.addEventListener("resize", () => {
  const rows = byId("benchmark-table").querySelectorAll("tr");
  if (rows.length) loadBenchmark();
});
checkApi();
loadBenchmark();
