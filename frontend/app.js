"use strict";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);
const SEV = ["low", "medium", "high", "critical"];

let lastDetections = [];
let charts = {};
let map = null;
let mapLayer = null;
let webcamStream = null;
let webcamTimer = null;

/* ============================ Tabs ============================ */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === `panel-${name}`)
  );
  if (name === "analytics") loadAnalytics();
  if (name === "map") loadMap();
  if (name === "history") loadHistory();
}

/* ============================ Health ============================ */
async function checkHealth() {
  const dot = $("status-dot");
  const text = $("status-text");
  const badge = $("mode-badge");
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const fallback = d.detector_fallback || d.using_fallback || d.engine === "opencv";
    dot.className = "dot " + (fallback ? "dot-warn" : "dot-on");
    text.textContent = d.status === "healthy" || d.status === "ok" ? "online" : (d.status || "online");
    badge.classList.remove("hidden");
    badge.textContent = fallback ? "OpenCV heuristic mode" : "YOLOv11 model";
  } catch {
    dot.className = "dot dot-off";
    text.textContent = "offline";
  }
}

/* ============================ Helpers ============================ */
function severityClass(level) {
  const l = String(level || "").toLowerCase();
  return SEV.includes(l) ? `sev sev-${l}` : "sev";
}
function scoreColor(s) {
  if (s == null) return "#3da9fc";
  if (s >= 80) return "var(--low)";
  if (s >= 60) return "var(--medium)";
  if (s >= 40) return "var(--high)";
  return "var(--critical)";
}
function metric(label, value, sub) {
  return `<div class="metric"><div class="label">${label}</div>
    <div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}
function money(v, cur) {
  if (v == null) return "—";
  return `${cur || "$"}${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

/* ============================ DETECT ============================ */
const dropzone = $("dropzone");
const fileInput = $("file-input");
let currentFile = null;

$("browse-btn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(file) {
  if (!file.type.startsWith("image/")) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const p = $("preview");
    p.src = e.target.result;
    p.classList.remove("hidden");
    document.querySelector(".dz-inner").classList.add("hidden");
  };
  reader.readAsDataURL(file);
  $("analyze-btn").disabled = false;
}

$("use-geo").addEventListener("change", (e) => {
  if (e.target.checked && navigator.geolocation) {
    navigator.geolocation.getCurrentPosition((pos) => {
      $("lat").value = pos.coords.latitude.toFixed(6);
      $("lng").value = pos.coords.longitude.toFixed(6);
    });
  }
});

$("analyze-btn").addEventListener("click", async () => {
  if (!currentFile) return;
  const btn = $("analyze-btn");
  btn.disabled = true;
  $("spinner").classList.remove("hidden");
  $("error").classList.add("hidden");
  try {
    const fd = new FormData();
    fd.append("file", currentFile);
    fd.append("annotate", $("annotate").checked);
    if ($("lat").value) fd.append("latitude", $("lat").value);
    if ($("lng").value) fd.append("longitude", $("lng").value);
    const r = await fetch(`${API}/detect/image`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    renderResult(await r.json());
  } catch (err) {
    $("error").textContent = "⚠️ " + err.message;
    $("error").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    $("spinner").classList.add("hidden");
  }
});

function renderResult(data) {
  $("results").classList.remove("hidden");
  lastDetections = data.detections || [];
  const score = data.road_condition_score;
  $("metrics").innerHTML =
    metric("Anomalies", data.count ?? lastDetections.length) +
    metric("Road Score", score != null ? `${score}` : "—", "0 = worst · 100 = best") +
    metric("Repair Cost", money(data.estimated_repair_cost, data.currency === "USD" ? "$" : data.currency + " "), "estimated") +
    metric("Latency", data.processing_time_ms != null ? `${Math.round(data.processing_time_ms)} ms` : "—") +
    metric("Engine", (data.model_version || "—").includes("opencv") ? "OpenCV" : "YOLOv11", data.model_version || "");

  if (data.annotated_image_base64) {
    $("result-image").src = "data:image/jpeg;base64," + data.annotated_image_base64;
    $("result-image").classList.remove("hidden");
  } else {
    $("result-image").classList.add("hidden");
  }

  const tbody = document.querySelector("#anomaly-table tbody");
  tbody.innerHTML = "";
  $("anomaly-count").textContent = lastDetections.length;
  if (lastDetections.length === 0) {
    $("no-anomalies").classList.remove("hidden");
    $("download-csv").classList.add("hidden");
  } else {
    $("no-anomalies").classList.add("hidden");
    $("download-csv").classList.remove("hidden");
    lastDetections.forEach((d, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${i + 1}</td>
        <td>${d.class_name}</td>
        <td>${(d.confidence * 100).toFixed(0)}%</td>
        <td><span class="${severityClass(d.severity_level)}">${d.severity_level}</span></td>
        <td>${(d.urgency || "").replace(/_/g, " ")}</td>
        <td>${d.depth_mm != null ? d.depth_mm.toFixed(0) + " mm" : "—"}</td>
        <td>${d.estimated_repair_cost != null ? money(d.estimated_repair_cost, "$") : "—"}</td>`;
      tbody.appendChild(tr);
    });
  }
  renderLegend();
}

function renderLegend() {
  const counts = {};
  lastDetections.forEach((d) => {
    const l = String(d.severity_level || "").toLowerCase();
    counts[l] = (counts[l] || 0) + 1;
  });
  $("severity-legend").innerHTML = SEV.filter((l) => counts[l])
    .map((l) => `<span class="sev sev-${l}">${l}: ${counts[l]}</span>`).join("");
}

$("download-csv").addEventListener("click", () => {
  const cols = ["#", "type", "confidence", "severity", "urgency", "depth_mm", "cost"];
  const rows = lastDetections.map((d, i) =>
    [i + 1, d.class_name, d.confidence, d.severity_level, d.urgency, d.depth_mm ?? "", d.estimated_repair_cost ?? ""]);
  const csv = [cols.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "anomalies.csv";
  a.click();
});

/* ============================ BATCH ============================ */
$("batch-input").addEventListener("change", () => {
  $("batch-btn").disabled = $("batch-input").files.length === 0;
});
$("batch-btn").addEventListener("click", async () => {
  const files = $("batch-input").files;
  if (!files.length) return;
  $("batch-btn").disabled = true;
  $("batch-spinner").classList.remove("hidden");
  $("batch-error").classList.add("hidden");
  try {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    fd.append("annotate", "true");
    const r = await fetch(`${API}/detect/batch`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    renderBatch(await r.json());
  } catch (err) {
    $("batch-error").textContent = "⚠️ " + err.message;
    $("batch-error").classList.remove("hidden");
  } finally {
    $("batch-btn").disabled = false;
    $("batch-spinner").classList.add("hidden");
  }
});

function renderBatch(d) {
  $("batch-results").classList.remove("hidden");
  $("batch-metrics").innerHTML =
    metric("Images", d.total_images, `${d.succeeded} ok · ${d.failed} failed`) +
    metric("Total Anomalies", d.total_anomalies) +
    metric("Avg Road Score", d.avg_road_score) +
    metric("Total Cost", money(d.total_repair_cost, "$"));
  $("batch-grid").innerHTML = d.items.map((it) => {
    if (it.error) {
      return `<div class="batch-item"><div class="meta">
        <div class="fname">${it.filename}</div>
        <div class="error" style="margin:0">⚠️ ${it.error}</div></div></div>`;
    }
    const img = it.annotated_image_base64
      ? `<img src="data:image/jpeg;base64,${it.annotated_image_base64}" />` : "";
    return `<div class="batch-item">${img}<div class="meta">
      <div class="fname">${it.filename}</div>
      <div class="row"><span>${it.count} anomalies</span><span>${money(it.estimated_repair_cost, "$")}</span></div>
      <div class="score-bar"><span style="width:${it.road_condition_score ?? 0}%;background:${scoreColor(it.road_condition_score)}"></span></div>
      </div></div>`;
  }).join("");
}

/* ============================ VIDEO ============================ */
$("video-input").addEventListener("change", () => {
  $("video-btn").disabled = $("video-input").files.length === 0;
});
$("video-btn").addEventListener("click", async () => {
  const file = $("video-input").files[0];
  if (!file) return;
  $("video-btn").disabled = true;
  $("video-spinner").classList.remove("hidden");
  $("video-error").classList.add("hidden");
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("max_frames", $("max-frames").value || "40");
    const r = await fetch(`${API}/detect/video/sync`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    renderVideo(await r.json());
  } catch (err) {
    $("video-error").textContent = "⚠️ " + err.message;
    $("video-error").classList.remove("hidden");
  } finally {
    $("video-btn").disabled = false;
    $("video-spinner").classList.add("hidden");
  }
});

function renderVideo(d) {
  $("video-results").classList.remove("hidden");
  $("video-metrics").innerHTML =
    metric("Frames", d.total_frames, `${d.processed_frames} processed`) +
    metric("Unique Anomalies", d.unique_anomalies, `${d.total_detections} total`) +
    metric("Avg Road Score", d.avg_road_score) +
    metric("Repair Cost", money(d.estimated_repair_cost, "$")) +
    metric("Time", `${Math.round(d.processing_time_ms)} ms`);
  $("video-frames").innerHTML = (d.sample_frames || []).map((f) =>
    `<div class="batch-item">
      <img src="data:image/jpeg;base64,${f.annotated_image_base64}" />
      <div class="meta"><div class="row"><span>frame ${f.frame_index}</span><span>${f.count} found</span></div></div>
    </div>`).join("") || '<p class="muted">No anomalies detected in sampled frames.</p>';
}

/* ============================ WEBCAM ============================ */
$("webcam-start").addEventListener("click", startWebcam);
$("webcam-stop").addEventListener("click", stopWebcam);

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    $("webcam").srcObject = webcamStream;
    $("webcam-start").classList.add("hidden");
    $("webcam-stop").classList.remove("hidden");
    $("webcam-result").classList.remove("hidden");
    webcamTimer = setInterval(captureWebcam, 2500);
  } catch (err) {
    $("webcam-error").textContent = "⚠️ Cannot access camera: " + err.message;
    $("webcam-error").classList.remove("hidden");
  }
}

function stopWebcam() {
  if (webcamTimer) clearInterval(webcamTimer);
  if (webcamStream) webcamStream.getTracks().forEach((t) => t.stop());
  webcamStream = null;
  $("webcam-start").classList.remove("hidden");
  $("webcam-stop").classList.add("hidden");
}

async function captureWebcam() {
  const video = $("webcam");
  if (!video.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.8));
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("annotate", "true");
  fd.append("persist", $("webcam-persist").checked ? "true" : "false");
  try {
    const r = await fetch(`${API}/detect/image`, { method: "POST", body: fd });
    if (!r.ok) return;
    const d = await r.json();
    if (d.annotated_image_base64)
      $("webcam-result").src = "data:image/jpeg;base64," + d.annotated_image_base64;
    $("webcam-metrics").innerHTML =
      metric("Anomalies", d.count) +
      metric("Road Score", d.road_condition_score ?? "—") +
      metric("Cost", money(d.estimated_repair_cost, "$"));
  } catch { /* ignore transient errors */ }
}

/* ============================ MAP ============================ */
async function loadMap() {
  if (!map) {
    map = L.map("map").setView([20, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap", maxZoom: 19,
    }).addTo(map);
  }
  setTimeout(() => map.invalidateSize(), 200);
  await refreshMap();
}
$("map-refresh").addEventListener("click", refreshMap);

async function refreshMap() {
  try {
    const r = await fetch(`${API}/analytics/geojson`);
    const gj = await r.json();
    if (mapLayer) map.removeLayer(mapLayer);
    const feats = gj.features || [];
    $("map-empty").classList.toggle("hidden", feats.length > 0);
    if (!feats.length) return;
    mapLayer = L.layerGroup();
    const bounds = [];
    feats.forEach((f) => {
      const [lng, lat] = f.geometry.coordinates;
      const p = f.properties;
      const color = scoreColorHex(p.road_condition_score);
      L.circleMarker([lat, lng], {
        radius: 6 + Math.min(14, (p.anomaly_count || 1) * 2),
        color, fillColor: color, fillOpacity: 0.6, weight: 2,
      }).bindPopup(
        `<b>${p.anomaly_count} anomalies</b><br>Road score: ${p.road_condition_score ?? "—"}<br>` +
        `Cost: ${money(p.estimated_repair_cost, "$")}<br><small>${p.source} · ${new Date(p.created_at).toLocaleString()}</small>`
      ).addTo(mapLayer);
      bounds.push([lat, lng]);
    });
    mapLayer.addTo(map);
    if (bounds.length) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 16 });
  } catch (err) {
    console.error(err);
  }
}
function scoreColorHex(s) {
  if (s == null) return "#3da9fc";
  if (s >= 80) return "#4ade80";
  if (s >= 60) return "#fbbf24";
  if (s >= 40) return "#fb923c";
  return "#ef4444";
}

/* ============================ ANALYTICS ============================ */
$("analytics-refresh").addEventListener("click", loadAnalytics);

async function loadAnalytics() {
  let s, timeline;
  try {
    [s, timeline] = await Promise.all([
      fetch(`${API}/analytics/summary`).then((r) => r.json()),
      fetch(`${API}/analytics/timeline`).then((r) => r.json()),
    ]);
  } catch { return; }

  $("analytics-metrics").innerHTML =
    metric("Detections", s.total_detections) +
    metric("Anomalies", s.total_anomalies, `${s.critical_count} critical`) +
    metric("Avg Road Score", s.avg_road_score) +
    metric("Avg Confidence", `${Math.round((s.avg_confidence || 0) * 100)}%`) +
    metric("Total Repair Cost", money(s.total_repair_cost, "$")) +
    metric("Geotagged", s.geotagged_count);

  const sevColors = { LOW: "#4ade80", MEDIUM: "#fbbf24", HIGH: "#fb923c", CRITICAL: "#ef4444" };
  drawChart("chart-severity", "doughnut",
    Object.keys(s.by_severity), Object.values(s.by_severity),
    Object.keys(s.by_severity).map((k) => sevColors[k] || "#3da9fc"));
  drawChart("chart-class", "bar",
    Object.keys(s.by_class), Object.values(s.by_class), "#3da9fc");
  drawChart("chart-urgency", "polarArea",
    Object.keys(s.by_urgency).map((u) => u.replace(/_/g, " ")), Object.values(s.by_urgency),
    ["#4ade80", "#fbbf24", "#fb923c", "#ef4444"]);
  drawLine("chart-timeline", timeline);
}

function drawChart(id, type, labels, data, colors) {
  const ctx = $(id);
  if (charts[id]) charts[id].destroy();
  if (!labels.length) { ctx.getContext("2d").clearRect(0, 0, ctx.width, ctx.height); return; }
  charts[id] = new Chart(ctx, {
    type,
    data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: "#0c1018", borderWidth: 1 }] },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#8a99ad" }, display: type !== "bar" } },
      scales: type === "bar" ? {
        x: { ticks: { color: "#8a99ad" }, grid: { color: "#25334a" } },
        y: { ticks: { color: "#8a99ad" }, grid: { color: "#25334a" }, beginAtZero: true },
      } : {},
    },
  });
}

function drawLine(id, timeline) {
  const ctx = $(id);
  if (charts[id]) charts[id].destroy();
  const labels = timeline.map((_, i) => i + 1);
  const data = timeline.map((t) => t.road_condition_score);
  charts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Road score", data, borderColor: "#3da9fc",
        backgroundColor: "rgba(61,169,252,.15)", fill: true, tension: 0.3, pointRadius: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#8a99ad" } } },
      scales: {
        x: { ticks: { color: "#8a99ad" }, grid: { color: "#25334a" } },
        y: { ticks: { color: "#8a99ad" }, grid: { color: "#25334a" }, min: 0, max: 100 },
      },
    },
  });
}

/* ============================ HISTORY ============================ */
$("history-refresh").addEventListener("click", loadHistory);
$("history-source").addEventListener("change", loadHistory);
$("history-clear").addEventListener("click", async () => {
  if (!confirm("Delete all stored detections? This cannot be undone.")) return;
  await fetch(`${API}/analytics/history`, { method: "DELETE" });
  loadHistory();
});

async function loadHistory() {
  const src = $("history-source").value;
  let rows;
  try {
    const url = `${API}/analytics/history?limit=60&include_thumbnail=true` + (src ? `&source=${src}` : "");
    rows = await fetch(url).then((r) => r.json());
  } catch { return; }
  $("history-empty").classList.toggle("hidden", rows.length > 0);
  $("history-grid").innerHTML = rows.map((d) => {
    const img = d.thumbnail
      ? `<img src="data:image/jpeg;base64,${d.thumbnail}" />`
      : `<img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E" />`;
    return `<div class="hist-card">${img}<div class="body">
      <div class="src">${d.source} · ${new Date(d.created_at).toLocaleString()}</div>
      <div class="row"><span>${d.anomaly_count} anomalies</span><span>${money(d.estimated_repair_cost, "$")}</span></div>
      <div class="row"><span class="muted">Road score</span><span>${d.road_condition_score ?? "—"}</span></div>
      <div class="score-bar"><span style="width:${d.road_condition_score ?? 0}%;background:${scoreColor(d.road_condition_score)}"></span></div>
      </div></div>`;
  }).join("");
}

/* ============================ REPORTS ============================ */
$("report-btn").addEventListener("click", async () => {
  $("report-spinner").classList.remove("hidden");
  $("report-error").classList.add("hidden");
  $("report-ok").classList.add("hidden");
  try {
    const title = encodeURIComponent($("report-title").value || "Road Maintenance Report");
    const r = await fetch(`${API}/reports/offline/generate?title=${title}`, { method: "POST" });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "road_maintenance_report.pdf";
    a.click();
    $("report-ok").classList.remove("hidden");
  } catch (err) {
    $("report-error").textContent = "⚠️ " + err.message;
    $("report-error").classList.remove("hidden");
  } finally {
    $("report-spinner").classList.add("hidden");
  }
});

/* ============================ Init ============================ */
checkHealth();
setInterval(checkHealth, 15000);
