const state = {
  catalog: null,
  trainingProfiles: null,
  openProfileCards: new Set(),
  family: "RoboCasa-Kitchen Ckpt",
  run: null,
  runManifest: null,
  sequences: null,
  activeRunId: null,
  runManifestsById: new Map(),
  sequencesByRunId: new Map(),
  pointsByChart: new Map(),
  actionChunks: null,
  actionChunksLoading: false,
  actionChunksError: null,
  visibleActionDims: null,
  selectedCharts: [],
  preferredFeatures: [],
  cam: "left_view",
  selectionMode: "single",
  selectedPoints: [],
  selected: null,
  visibleSeqs: new Set(),
  openTasks: new Set(),
  openTaskGroups: new Set(["Cabinet", "Coffee", "Pick & Place", "Appliances", "Sink"]),
  lastHash: "",
  syncingVideos: false,
  videoMode: "chunk",
  videoClockGeneration: 0,
  panelWidth: 360,
  panMode: false,
  lassoMode: false,
  lassoRegions: [],
  colorMode: "task",
  timestepRange: [0, 1],
  spacingMode: "original",
  hoveredEpisodeKey: null,
  chartStates: new Map(),
  pendingPanelScrollTop: null,
  pendingTaskPanelScrollTop: null,
};

function chartKey(runId, feature) {
  return `${runId}::${feature}`;
}

function episodeKey(runId, seqId) {
  return `${runId}::${seqId}`;
}

function getSequencesForRun(runId) {
  return state.sequencesByRunId.get(runId) || state.sequences;
}

function getSequenceById(runId, seqId) {
  return getSequencesForRun(runId)?.sequences?.[seqId] || null;
}

function getEpisodeSelectionKey(runId, seqId) {
  const seq = getSequenceById(runId, seqId);
  if (!seq) return null;
  const dataset = seq.dataset_rel_path || seq.task_name || "";
  return `${dataset}::${seq.episode_index}`;
}

function getPointSelectionKey(runId, point) {
  const episodeSelectionKey = getEpisodeSelectionKey(runId, point.seq);
  const frame = numericValue(point.frame);
  if (!episodeSelectionKey || frame === null) return null;
  return `${episodeSelectionKey}::${frame}`;
}

function getPointGlobalKey(runId, point) {
  const frame = numericValue(point.frame);
  if (frame === null) return null;
  return `${point.seq}::${frame}`;
}

function buildSelection(runId, point, feature = null) {
  return {
    runId,
    feature: feature || point.feature || null,
    seq: point.seq,
    anchor: point.anchor,
    frame: point.frame,
    progress: point.progress,
    episodeSelectionKey: getEpisodeSelectionKey(runId, point.seq),
    selectionKey: getPointSelectionKey(runId, point),
  };
}

function logicalSelectionKey(selection) {
  return selection?.selectionKey
    || `${selection?.runId}::${selection?.seq}::${selection?.frame}`;
}

function selectionKey(selection) {
  if (!selection) return "";
  return `${selection.runId}::${selection.feature || ""}::${selection.seq}::${selection.anchor}::${selection.frame}`;
}

function setSelectedPoints(points) {
  state.selectedPoints = points;
  state.selected = points[points.length - 1] || null;
}

function clearSelectedPoints() {
  setSelectedPoints([]);
}

function removeSelectedPointsForRun(runId) {
  setSelectedPoints(state.selectedPoints.filter((point) => point.runId !== runId));
}

function isSelectionVisible(selection) {
  const seq = getSequenceById(selection.runId, selection.seq);
  return isSequenceVisibleForRun(selection.runId, selection.seq)
    && isPointInTimestepRange(selection, seq);
}

function getSelectionIndex(selection) {
  const key = logicalSelectionKey(selection);
  return state.selectedPoints.findIndex((point) => logicalSelectionKey(point) === key);
}

function getSelectionAccent(selection) {
  const index = getSelectionIndex(selection);
  if (state.selectionMode !== "multi" || index < 0) return "#facc15";
  return selectionPalette[index % selectionPalette.length];
}

function getRunById(runId) {
  return state.catalog?.runs?.find((run) => run.id === runId) || null;
}

async function loadActionChunks() {
  if (state.actionChunks || state.actionChunksLoading) return;
  state.actionChunksLoading = true;
  state.actionChunksError = null;
  try {
    const payload = await fetchJson("./data/action_chunks/official_manifest_transformed_actions.json");
    const byPoint = new Map();
    (payload.seq_ids || []).forEach((seqId, index) => {
      const frame = Number(payload.frame_indices?.[index] ?? 0);
      byPoint.set(`${Number(seqId)}::${frame}`, {
        seqId: Number(seqId),
        anchor: Number(payload.anchor_ids?.[index] ?? 0),
        frame,
        values: payload.chunks?.[index] || [],
      });
    });
    state.actionChunks = { ...payload, byPoint };
  } catch (error) {
    state.actionChunksError = error;
  } finally {
    state.actionChunksLoading = false;
  }
}

function getActionChunk(selection) {
  return state.actionChunks?.byPoint?.get(`${Number(selection?.seq)}::${Number(selection?.frame)}`) || null;
}

function actionDimLabels(dimCount) {
  const labelsByKey = {
    "action.end_effector_position": ["eef pos x", "eef pos y", "eef pos z"],
    "action.end_effector_rotation": ["eef rot x", "eef rot y", "eef rot z"],
    "action.gripper_close": ["gripper"],
    "action.base_motion": ["base 0", "base 1", "base 2", "base 3"],
    "action.control_mode": ["control"],
  };
  const labels = [];
  for (const key of state.actionChunks?.action_keys || []) {
    labels.push(...(labelsByKey[key] || [key.replace(/^action\./, "")]));
  }
  while (labels.length < dimCount) labels.push(`dim ${labels.length}`);
  return labels.slice(0, dimCount);
}

function visibleActionDims(dimCount) {
  if (!state.visibleActionDims) {
    state.visibleActionDims = new Set(Array.from({ length: dimCount }, (_, index) => index));
  }
  return state.visibleActionDims;
}

function toggleActionDim(dim, dimCount) {
  const visible = visibleActionDims(dimCount);
  if (visible.has(dim)) visible.delete(dim);
  else visible.add(dim);
  renderPanel();
}

const familyOrder = ["RoboCasa-Kitchen Ckpt", "Action (TimewarpVAE)", "Action (AE / PCA)"];
const featureOrder = ["raw", "processed", "action"];

const kitchenTaskGroups = ["Cabinet", "Coffee", "Pick & Place", "Appliances", "Sink"];
const kitchenTaskOrder = [
  "CloseDoubleDoor", "CloseDrawer", "CloseSingleDoor", "OpenDoubleDoor", "OpenDrawer", "OpenSingleDoor",
  "CoffeePressButton", "CoffeeServeMug", "CoffeeSetupMug",
  "PnPCabToCounter", "PnPCounterToCab", "PnPCounterToMicrowave", "PnPCounterToSink", "PnPCounterToStove",
  "PnPMicrowaveToCounter", "PnPSinkToCounter", "PnPStoveToCounter",
  "TurnOffMicrowave", "TurnOffStove", "TurnOnMicrowave", "TurnOnStove",
  "TurnOffSinkFaucet", "TurnOnSinkFaucet", "TurnSinkSpout",
];

function getTaskMeta(seqOrName) {
  if (typeof seqOrName === "object" && seqOrName) {
    return { group: seqOrName.task_group || "Other", label: seqOrName.task_label || seqOrName.task_name };
  }
  return { group: "Other", label: String(seqOrName || "") };
}

function orderIndex(list, value) {
  const index = list.indexOf(value);
  return index === -1 ? list.length : index;
}

function getCatalogRunIndex(runId) {
  const index = state.catalog?.runs?.findIndex((run) => run.id === runId) ?? -1;
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function chartSortKey(chart) {
  const run = getRunById(chart.runId);
  return {
    family: orderIndex(familyOrder, run?.family),
    runIndex: getCatalogRunIndex(chart.runId),
    runLabel: run?.label || chart.runId,
    feature: orderIndex(featureOrder, chart.feature),
    featureName: chart.feature,
  };
}

function sortSelectedCharts() {
  state.selectedCharts.sort((a, b) => {
    const ka = chartSortKey(a);
    const kb = chartSortKey(b);
    return ka.family - kb.family
      || ka.feature - kb.feature
      || ka.featureName.localeCompare(kb.featureName)
      || ka.runIndex - kb.runIndex
      || ka.runLabel.localeCompare(kb.runLabel);
  });
}

const familyLabels = {
  "RoboCasa-Kitchen Ckpt": "RoboCasa-Kitchen Ckpt",
  "Action (TimewarpVAE)": "Action (TimewarpVAE)",
  "Action (AE / PCA)": "Action (AE / PCA)",
};

function hashParams(extra = {}) {
  return new URLSearchParams(extra).toString();
}

function resetRunState() {
  state.run = null;
  state.runManifest = null;
  state.sequences = null;
  state.activeRunId = null;
  state.runManifestsById.clear();
  state.sequencesByRunId.clear();
  state.pointsByChart.clear();
  state.actionChunks = null;
  state.actionChunksLoading = false;
  state.actionChunksError = null;
  state.visibleActionDims = null;
  state.selectedCharts = [];
  state.preferredFeatures = [];
  clearSelectedPoints();
  state.visibleSeqs = new Set();
  state.openTasks = new Set();
  state.hoveredEpisodeKey = null;
  state.lassoRegions = [];
  state.chartStates.clear();
}

async function loadCatalog() {
  const pending = [];
  if (!state.catalog) {
    pending.push(fetchJson("./data/catalog.json").then((catalog) => {
      state.catalog = catalog;
    }));
  }
  if (!state.trainingProfiles) {
    pending.push(fetchJson("./data/training_profiles.json").then((payload) => {
      state.trainingProfiles = payload.profiles || {};
    }));
  }
  await Promise.all(pending);
}

const colors = [
  "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
  "#be123c", "#4f46e5", "#65a30d", "#c026d3", "#0f766e", "#b45309",
  "#1d4ed8", "#b91c1c", "#15803d", "#7c3aed", "#d97706", "#0e7490",
  "#9f1239", "#4338ca", "#4d7c0f", "#a21caf", "#115e59", "#92400e",
  "#0369a1", "#a16207"
];

const timestepPalette = ["#173b66", "#2f6f8f", "#82a782", "#f3df58"];
const selectionPalette = ["#f9a8d4", "#93c5fd", "#86efac"];
const lassoPalette = [
  { fill: "rgba(147, 197, 253, 0.34)", stroke: "rgba(59, 130, 246, 0.68)" },
  { fill: "rgba(249, 168, 212, 0.34)", stroke: "rgba(219, 39, 119, 0.64)" },
  { fill: "rgba(134, 239, 172, 0.34)", stroke: "rgba(22, 163, 74, 0.62)" },
  { fill: "rgba(253, 230, 138, 0.38)", stroke: "rgba(217, 119, 6, 0.66)" },
  { fill: "rgba(196, 181, 253, 0.34)", stroke: "rgba(124, 58, 237, 0.62)" },
  { fill: "rgba(125, 211, 252, 0.34)", stroke: "rgba(2, 132, 199, 0.64)" },
  { fill: "rgba(252, 165, 165, 0.34)", stroke: "rgba(220, 38, 38, 0.62)" },
  { fill: "rgba(167, 243, 208, 0.34)", stroke: "rgba(5, 150, 105, 0.62)" },
  { fill: "rgba(253, 186, 116, 0.34)", stroke: "rgba(234, 88, 12, 0.62)" },
  { fill: "rgba(216, 180, 254, 0.34)", stroke: "rgba(147, 51, 234, 0.62)" },
];
const softSpacingGamma = 0.55;
const spacingEpsilon = 1e-9;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16),
  ];
}

function rgbToHex(rgb) {
  return `#${rgb.map((value) =>
    Math.round(value).toString(16).padStart(2, "0")
  ).join("")}`;
}

function complementColor(hex) {
  if (!/^#[0-9a-fA-F]{6}$/.test(hex || "")) return "#22d3ee";
  return rgbToHex(hexToRgb(hex).map((value) => 255 - value));
}

function interpolateColor(startHex, endHex, t) {
  const start = hexToRgb(startHex);
  const end = hexToRgb(endHex);
  return rgbToHex(start.map((value, i) => value + (end[i] - value) * t));
}

function interpolatePalette(palette, t) {
  const scaled = clamp(t, 0, 1) * (palette.length - 1);
  const index = Math.min(palette.length - 2, Math.floor(scaled));
  return interpolateColor(palette[index], palette[index + 1], scaled - index);
}

function numericValue(value) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function getPointTimestep(point) {
  return numericValue(point.frame) ?? numericValue(point.anchor);
}

function getPointTimestepProgress(point, seq) {
  const frame = getPointTimestep(point);
  const episodeLength = numericValue(seq?.episode_length);
  if (frame !== null && episodeLength !== null && episodeLength > 1) {
    return clamp(frame / (episodeLength - 1), 0, 1);
  }
  const progress = numericValue(point.progress);
  if (progress !== null) return clamp(progress, 0, 1);
  return null;
}

function getTimestepColor(point, seq) {
  const t = getPointTimestepProgress(point, seq);
  if (t === null) return timestepPalette[0];
  return interpolatePalette(timestepPalette, t);
}

function getPointColor(point, seq) {
  if (state.colorMode === "timestep") return getTimestepColor(point, seq);
  return getTaskColor(seq);
}

function isPointInTimestepRange(point, seq) {
  const t = getPointTimestepProgress(point, seq);
  if (t === null) return true;
  return t >= state.timestepRange[0] && t <= state.timestepRange[1];
}

function getTaskColor(seq) {
  const taskId = Number.isFinite(seq?.task_id) ? seq.task_id : 0;
  return colors[taskId % colors.length] || colors[0];
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function getTemporalStepDistances(points) {
  const distances = [];
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const distance = Math.hypot(curr.x - prev.x, curr.y - prev.y);
    if (distance > spacingEpsilon) distances.push(distance);
  }
  return distances;
}

function getPathLength(points) {
  return getTemporalStepDistances(points).reduce((sum, distance) => sum + distance, 0);
}

function preserveEpisodePathScale(originalPoints, adjustedPoints) {
  const originalLength = getPathLength(originalPoints);
  const adjustedLength = getPathLength(adjustedPoints);
  if (originalLength <= spacingEpsilon || adjustedLength <= spacingEpsilon) return adjustedPoints;
  const scale = originalLength / adjustedLength;
  const anchor = adjustedPoints[0];
  return adjustedPoints.map((point) => ({
    ...point,
    x: anchor.x + (point.x - anchor.x) * scale,
    y: anchor.y + (point.y - anchor.y) * scale,
  }));
}

function getAdjustedStepDistance(distance, referenceDistance) {
  if (distance <= spacingEpsilon || referenceDistance <= spacingEpsilon) return distance;
  if (state.spacingMode === "equal") return referenceDistance;
  if (state.spacingMode === "soft") {
    return referenceDistance * Math.pow(distance / referenceDistance, softSpacingGamma);
  }
  return distance;
}

function applyTemporalSpacing(points) {
  if (state.spacingMode === "original") return points;
  const bySeq = new Map();
  for (const point of points) {
    if (!bySeq.has(point.seq)) bySeq.set(point.seq, []);
    bySeq.get(point.seq).push(point);
  }

  const adjustedByPoint = new Map();
  for (const seqPoints of bySeq.values()) {
    seqPoints.sort((a, b) => a.anchor - b.anchor);
    const referenceDistance = median(getTemporalStepDistances(seqPoints));
    const adjustedSeqPoints = [];
    if (seqPoints.length) adjustedSeqPoints.push({ ...seqPoints[0] });

    for (let i = 1; i < seqPoints.length; i++) {
      const prevOriginal = seqPoints[i - 1];
      const currOriginal = seqPoints[i];
      const prevAdjusted = adjustedSeqPoints[i - 1];
      const dx = currOriginal.x - prevOriginal.x;
      const dy = currOriginal.y - prevOriginal.y;
      const distance = Math.hypot(dx, dy);
      if (!prevAdjusted || referenceDistance <= spacingEpsilon) {
        adjustedSeqPoints.push({ ...currOriginal });
        continue;
      }
      if (distance <= spacingEpsilon) {
        adjustedSeqPoints.push({
          ...currOriginal,
          x: prevAdjusted.x,
          y: prevAdjusted.y,
        });
        continue;
      }
      const adjustedDistance = getAdjustedStepDistance(distance, referenceDistance);
      adjustedSeqPoints.push({
        ...currOriginal,
        x: prevAdjusted.x + (dx / distance) * adjustedDistance,
        y: prevAdjusted.y + (dy / distance) * adjustedDistance,
      });
    }
    const scaledSeqPoints = preserveEpisodePathScale(seqPoints, adjustedSeqPoints);
    seqPoints.forEach((point, index) => {
      adjustedByPoint.set(point, scaledSeqPoints[index]);
    });
  }
  return points.map((point) => adjustedByPoint.get(point) || point);
}

function sequenceKey(seq) {
  const dataset = seq.dataset_rel_path || "";
  return `${dataset}\u0000${seq.episode_index}`;
}

function getVisibleSequenceKeys() {
  const sequences = state.sequences?.sequences || [];
  const keys = new Set();
  for (const seq of sequences) {
    if (state.visibleSeqs.has(seq.seq_id)) keys.add(sequenceKey(seq));
  }
  return keys;
}

function isSequenceVisibleForRun(runId, seqId, visibleKeys = getVisibleSequenceKeys()) {
  if (runId === state.activeRunId) return state.visibleSeqs.has(seqId);
  const sequences = state.sequencesByRunId.get(runId);
  const seq = sequences?.sequences?.[seqId];
  if (!seq) return false;
  return visibleKeys.has(sequenceKey(seq));
}

async function fetchJson(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

async function init() {
  renderShell();
  attachPanelResize();
  try {
    await loadCatalog();
    await applyHash();
    window.addEventListener("hashchange", applyHash);
    window.setInterval(() => {
      if (location.hash !== state.lastHash) applyHash();
    }, 150);
  } catch (err) {
    document.querySelector(".stage").innerHTML = `<p class="status error">${err.message}</p>`;
  }
}

async function applyHash() {
  state.lastHash = location.hash;
  const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
  if (hash.has("version")) {
    hash.delete("version");
    const cleanHash = hash.toString();
    history.replaceState(null, "", `${location.pathname}${location.search}${cleanHash ? `#${cleanHash}` : ""}`);
    state.lastHash = location.hash;
  }
  await loadCatalog();
  if (!state.catalog) return;
  const family = hash.get("family") || "RoboCasa-Kitchen Ckpt";
  const runId = hash.get("run");
  state.family = family;
  renderSidebar();
  if (runId) {
    const run = state.catalog.runs.find((r) => r.id === runId);
    if (run && (!state.run || state.run.id !== run.id)) {
      state.family = run.family;
      await loadRun(run, { updateHash: false });
      return;
    }
  }
  if (!runId) {
    if (state.selectedCharts.length) {
      if (!state.activeRunId && state.selectedCharts[0]) {
        state.activeRunId = state.selectedCharts[0].runId;
      }
      renderViewer();
      return;
    }
    state.run = null;
    state.runManifest = null;
    state.sequences = null;
    state.activeRunId = null;
    state.pointsByChart.clear();
    state.selectedCharts = [];
    state.chartStates.clear();
    clearSelectedPoints();
    state.visibleSeqs = new Set();
    state.openTasks = new Set();
    renderSidebar();
    renderEmpty();
  }
}

function renderShell() {
  document.getElementById("app").innerHTML = "";
  const shell = el("div", { class: "shell" }, [
    el("header", { class: "topbar" }, [
      el("div", { class: "title" }, [
        el("h1", { text: "RoboCasa-Kitchen temporal t-SNE" }),
        el("p", { text: "7,200 shared frame points from 240 episodes across six GR00T checkpoints and TimewarpVAE action embeddings." }),
      ]),
      el("a", { href: "https://github.com/minje227-coder/groot-robocasa-kitchen-tsne", text: "GitHub" }),
    ]),
    el("main", { class: "layout" }, [
      el("aside", { class: "sidebar" }),
      el("section", { class: "stage" }, [
        el("div", { class: "chart-wrap" }, [
          el("div", { class: "status", text: "Select a run." }),
        ]),
      ]),
      el("div", {
        class: "panel-resizer",
        role: "separator",
        "aria-orientation": "vertical",
        "aria-label": "Resize details panel",
        tabindex: "0",
      }),
      el("aside", { class: "panel" }),
    ]),
  ]);
  document.getElementById("app").appendChild(shell);
  applyPanelWidth();
}

function applyPanelWidth() {
  const layout = document.querySelector(".layout");
  if (!layout) return;
  layout.style.setProperty("--panel-width", `${state.panelWidth}px`);
}

function attachPanelResize() {
  const layout = document.querySelector(".layout");
  const handle = document.querySelector(".panel-resizer");
  if (!layout || !handle) return;

  const minWidth = 300;
  const maxWidth = 720;

  const updateWidthFromClientX = (clientX) => {
    const bounds = layout.getBoundingClientRect();
    const next = Math.min(maxWidth, Math.max(minWidth, bounds.right - clientX));
    state.panelWidth = next;
    applyPanelWidth();
  };

  handle.addEventListener("pointerdown", (event) => {
    if (window.innerWidth <= 1100) return;
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    document.body.classList.add("resizing-panel");

    const onMove = (moveEvent) => updateWidthFromClientX(moveEvent.clientX);
    const onUp = () => {
      document.body.classList.remove("resizing-panel");
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };

    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  });

  handle.addEventListener("keydown", (event) => {
    if (window.innerWidth <= 1100) return;
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? 24 : -24;
    state.panelWidth = Math.min(maxWidth, Math.max(minWidth, state.panelWidth + delta));
    applyPanelWidth();
  });
}

function renderActiveModelDetails() {
  const activeRunIds = [...new Set(state.selectedCharts.map((chart) => chart.runId))];
  const activeSet = new Set(activeRunIds);
  for (const runId of [...state.openProfileCards]) {
    if (!activeSet.has(runId)) state.openProfileCards.delete(runId);
  }

  const cards = activeRunIds.flatMap((runId) => {
    const run = getRunById(runId);
    const profile = state.trainingProfiles?.[runId];
    if (!run || !profile) return [];

    const isOpen = state.openProfileCards.has(runId);
    const features = state.selectedCharts
      .filter((chart) => chart.runId === runId)
      .map((chart) => chart.feature);

    const header = el("button", {
      class: "model-profile-toggle",
      "aria-expanded": String(isOpen),
      onclick: () => {
        if (state.openProfileCards.has(runId)) state.openProfileCards.delete(runId);
        else state.openProfileCards.add(runId);
        renderSidebar();
      },
    }, [
      el("span", { class: "model-profile-caret", text: isOpen ? "▾" : "▸" }),
      el("span", { class: "model-profile-title", text: run.label || run.id }),
      el("span", { class: "model-profile-features" }, features.map((feature) =>
        el("span", { class: "model-profile-feature", text: feature })
      )),
    ]);

    const facts = el("dl", { class: "model-profile-facts" },
      (profile.facts || []).flatMap(([label, value]) => [
        el("dt", { text: label }),
        el("dd", { text: value }),
      ])
    );

    const phases = el("div", { class: "phase-timeline" },
      (profile.phases || []).map((phase, index) => {
        const badgeClass = phase.enabled === true
          ? " on"
          : phase.enabled === false
            ? " off"
            : " neutral";
        return el("article", { class: "phase-card" }, [
          el("div", { class: "phase-head" }, [
            el("span", { class: "phase-index", text: "P" + (index + 1) }),
            el("div", { class: "phase-name-wrap" }, [
              el("strong", { class: "phase-name", text: phase.name }),
              el("span", { class: "phase-range", text: phase.range }),
            ]),
            el("span", { class: "phase-badge" + badgeClass, text: phase.badge }),
          ]),
          el("dl", { class: "phase-rows" },
            (phase.rows || []).flatMap(([label, value]) => [
              el("dt", { text: label }),
              el("dd", { text: value }),
            ])
          ),
        ]);
      })
    );

    const source = el("p", {
      class: "model-profile-source",
      text: "Source: " + (profile.source_label || "checkpoint metadata"),
      title: (profile.sources || []).join("\n"),
    });

    return [el("div", { class: "model-profile-card" + (isOpen ? " open" : "") }, [
      header,
      el("div", { class: "model-profile-body" + (isOpen ? " open" : "") }, [
        facts,
        el("h4", { class: "phase-heading", text: "Phase timeline" }),
        phases,
        source,
      ]),
    ])];
  });

  if (!cards.length) return null;
  return el("section", { class: "active-model-details" }, [
    el("h2", { text: "Active models" }),
    el("div", { class: "model-profile-list" }, cards),
  ]);
}

function renderSidebar() {
  const sidebar = document.querySelector(".sidebar");
  const families = [...(state.catalog.families || familyOrder)]
    .sort((a, b) => orderIndex(familyOrder, a) - orderIndex(familyOrder, b));
  const familyButtons = families.map((family) =>
    el("button", {
      class: `family-btn${family === state.family ? " active" : ""}`,
      text: familyLabels[family] || family,
      onclick: () => {
        state.family = family;
        location.hash = hashParams({ family });
        renderSidebar();
        if (state.selectedCharts.length) renderViewer();
        else renderEmpty();
      },
    })
  );
  const runs = (state.catalog.runs || []).filter((run) => run.family === state.family);
  const runButtons = runs.map((run) => {
    const isShown = state.selectedCharts.some((chart) => chart.runId === run.id);
    return el("div", {
      class: `run-btn${isShown ? " selected" : ""}`,
    }, [
      el("strong", { text: run.label || run.id }),
      renderRunFeatureToggle(run),
    ]);
  });

  sidebar.innerHTML = "";
  sidebar.appendChild(el("h2", { text: "Family" }));
  sidebar.appendChild(el("div", { class: "family-grid" }, familyButtons));
  sidebar.appendChild(el("h2", { text: "Runs" }));
  sidebar.appendChild(el("div", { class: "run-list" }, runButtons.length ? runButtons : [
    el("p", { class: "status", text: "No runs published yet." }),
  ]));
  sidebar.appendChild(el("div", { class: "sidebar-help" }, [
    el("p", { text: "Wheel: zoom" }),
    el("p", { text: "Wheel click + drag: pan" }),
  ]));
  const modelDetails = renderActiveModelDetails();
  if (modelDetails) sidebar.appendChild(modelDetails);
}

function renderEmpty() {
  document.querySelector(".stage").innerHTML = "";
  const stage = document.querySelector(".stage");
  stage.appendChild(el("div", { class: "home-panel" }, [
    el("h2", { text: "Frame t-SNE" }),
    el("p", { text: "Task-balanced manifest: 10 episodes per task and 30 uniformly sampled frames per episode." }),
    el("p", { class: "status", text: "Select a run from the sidebar." }),
  ]));
  document.querySelector(".panel").innerHTML = "";
}

function renderRunFeatureToggle(run) {
  return el("div", { class: "run-feature-toggle" }, (run.features || []).map((feature) =>
    el("button", {
      class: `mini-btn run-feature-btn${hasChart(run.id, feature) ? " active" : ""}`,
      text: feature,
      title: `${run.label || run.id} ${feature}`,
      onclick: (event) => toggleRunFeature(run, feature, event),
    })
  ));
}

async function toggleRunFeature(run, feature, event) {
  event.preventDefault();
  event.stopPropagation();
  if (state.activeRunId !== run.id) {
    await loadRun(run);
  }
  await toggleFeature(feature);
  renderSidebar();
  renderViewer();
}

async function loadRun(run, options = {}) {
  const isFirstChart = !state.selectedCharts.length;
  const previousVisibleKeys = getVisibleSequenceKeys();
  state.run = run;
  state.activeRunId = run.id;
  renderSidebar();
  if (!state.selectedCharts.length) {
    document.querySelector(".stage").innerHTML = `<div class="chart-wrap"><p class="status">Loading ${run.label || run.id}...</p></div>`;
    document.querySelector(".panel").innerHTML = "";
  }
  await loadRunAssets(run);
  if (isFirstChart) {
    state.visibleSeqs = new Set(state.sequences.sequences.map((seq) => seq.seq_id));
    state.openTasks = new Set();
  } else {
    state.visibleSeqs = new Set(
      state.sequences.sequences
        .filter((seq) => previousVisibleKeys.has(sequenceKey(seq)))
        .map((seq) => seq.seq_id)
    );
    if (!state.visibleSeqs.size) {
      state.visibleSeqs = new Set(state.sequences.sequences.map((seq) => seq.seq_id));
    }
  }
  state.preferredFeatures = getSelectedFeaturesForRun(run.id);
  if (options.updateHash !== false) {
    location.hash = hashParams({ family: run.family, run: run.id });
  }
  await Promise.all(state.selectedCharts.map((chart) => loadChartData(chart.runId, chart.feature)));
  ensureSelectedPoint(true);
  renderViewer();
}

async function loadRunAssets(run) {
  if (!state.runManifestsById.has(run.id)) {
    const base = `./${run.path}/`;
    const manifest = await fetchJson(`${base}manifest.json`);
    const sequences = await fetchJson(`${base}${manifest.sequences_file}`);
    state.runManifestsById.set(run.id, manifest);
    state.sequencesByRunId.set(run.id, sequences);
  }
  state.runManifest = state.runManifestsById.get(run.id);
  state.sequences = state.sequencesByRunId.get(run.id);
}

async function loadChartData(runId, feature) {
  const run = getRunById(runId);
  if (!run) return;
  await loadRunAssets(run);
  const key = chartKey(runId, feature);
  if (!state.pointsByChart.has(key)) {
    const base = `./${run.path}/`;
    const file = state.runManifestsById.get(runId).points_files[feature];
    state.pointsByChart.set(key, await fetchJson(`${base}${file}`));
  }
}

function getSelectedFeaturesForRun(runId) {
  return state.selectedCharts
    .filter((chart) => chart.runId === runId)
    .map((chart) => chart.feature);
}

function removeRunCharts(runId) {
  const removedCharts = state.selectedCharts.filter((chart) => chart.runId === runId);
  if (!removedCharts.length) return;
  state.selectedCharts = state.selectedCharts.filter((chart) => chart.runId !== runId);
  for (const chart of removedCharts) {
    state.chartStates.delete(chartKey(chart.runId, chart.feature));
  }
  removeSelectedPointsForRun(runId);
  if (!state.selectedCharts.length) {
    state.run = null;
    state.runManifest = null;
    state.sequences = null;
    state.activeRunId = null;
    state.preferredFeatures = [];
    state.visibleSeqs = new Set();
    state.openTasks = new Set();
    renderSidebar();
    renderEmpty();
    location.hash = hashParams({ family: state.family });
    return;
  }
  if (state.activeRunId === runId) {
    state.activeRunId = state.selectedCharts[0].runId;
  }
  const activeRun = getRunById(state.activeRunId);
  state.run = activeRun;
  state.runManifest = activeRun ? state.runManifestsById.get(activeRun.id) || null : null;
  state.sequences = activeRun ? state.sequencesByRunId.get(activeRun.id) || null : null;
  state.preferredFeatures = state.activeRunId ? getSelectedFeaturesForRun(state.activeRunId) : [];
  ensureSelectedPoint(true);
  renderSidebar();
  renderViewer();
  location.hash = hashParams({ family: state.family, run: state.activeRunId });
}

function removeChartSelection(runId, feature) {
  const hasSelection = state.selectedCharts.some((chart) => chart.runId === runId && chart.feature === feature);
  if (!hasSelection) return;
  if (state.selectedCharts.length === 1) {
    removeRunCharts(runId);
    return;
  }
  state.selectedCharts = state.selectedCharts.filter((chart) => !(chart.runId === runId && chart.feature === feature));
  state.chartStates.delete(chartKey(runId, feature));
  removeSelectedPointsForRun(runId);
  state.preferredFeatures = state.activeRunId ? getSelectedFeaturesForRun(state.activeRunId) : [];
  ensureSelectedPoint(true);
  renderSidebar();
  renderViewer();
}

function renderViewer() {
  sortSelectedCharts();
  ensureSelectedPoint(true);
  renderTabs();
  renderCharts();
  renderPanel();
}

function ensureSelectedPoint(allowReplace = false) {
  const selectedRunIsShown = state.selectedCharts.some((chart) =>
    chart.runId === state.selected?.runId && chart.feature === state.selected?.feature
  );
  const selectedSeq = state.selected ? getSequenceById(state.selected.runId, state.selected.seq) : null;
  if (
    state.selected
    && selectedRunIsShown
    && isSequenceVisibleForRun(state.selected.runId, state.selected.seq)
    && isPointInTimestepRange(state.selected, selectedSeq)
  ) return;
  if (state.selected && !allowReplace) return;
  const chart = state.selectedCharts[0];
  if (!chart) return;
  const pointPayload = state.pointsByChart.get(chartKey(chart.runId, chart.feature));
  if (!pointPayload || !pointPayload.points?.length) return;
  const visibleKeys = getVisibleSequenceKeys();
  const first = pointPayload.points.find((row) => {
    const point = { x: row[0], y: row[1], seq: row[2], anchor: row[3], frame: row[4], progress: row[5] };
    const seq = getSequenceById(chart.runId, point.seq);
    return isSequenceVisibleForRun(chart.runId, point.seq, visibleKeys)
      && isPointInTimestepRange(point, seq);
  });
  if (!first) {
    clearSelectedPoints();
    return;
  }
  setSelectedPoints([buildSelection(chart.runId, {
    seq: first[2],
    anchor: first[3],
    frame: first[4],
    progress: first[5],
  }, chart.feature)]);
}

function renderTabs() {
  state.videoClockGeneration += 1;
  const selectionControls = renderSelectionControls();
  const cameraControls = renderCameraControls();
  const colorControls = renderColorControls();
  const zoomControls = el("div", { class: "zoom-controls" }, [
    el("button", {
      class: `mini-btn zoom-btn pan-toggle${state.panMode ? " active" : ""}`,
      onclick: () => {
        state.panMode = !state.panMode;
        if (state.panMode) state.lassoMode = false;
        renderTabs();
        renderCharts();
      },
      title: "Toggle drag pan",
      "aria-label": "Toggle drag pan",
    }, [
      el("img", {
        class: "pan-icon",
        src: "./assets/move.png",
        alt: "",
        width: "44",
        height: "44",
        style: "width:44px;height:44px;",
      }),
    ]),
    renderLassoToolGroup(),
    el("button", {
      class: "mini-btn zoom-btn",
      text: "-",
      onclick: () => {
        scaleSelectedCharts(1 / 1.25);
      },
      title: "Zoom out",
    }),
    el("span", { class: "zoom-readout", text: getZoomReadout() }),
    el("button", {
      class: "mini-btn zoom-btn",
      text: "+",
      onclick: () => {
        scaleSelectedCharts(1.25);
      },
      title: "Zoom in",
    }),
  ]);
  const toolbar = el("div", { class: "stage-toolbar" }, [
    el("div", {
      class: "chart-controls-left",
    }, [
      selectionControls,
      ...(state.selectionMode === "multi" ? [cameraControls] : []),
    ]),
    el("div", {
      class: "chart-controls-right",
    }, [
      colorControls,
      zoomControls,
    ]),
  ]);
  document.querySelector(".stage").innerHTML = "";
  renderVideoStrip(document.querySelector(".stage"));
  document.querySelector(".stage").appendChild(toolbar);
  document.querySelector(".stage").appendChild(el("div", { class: "chart-grid" }));
}

function renderLassoToolGroup() {
  return el("div", { class: "lasso-tool-group" }, [
    el("button", {
      class: `mini-btn zoom-btn lasso-toggle${state.lassoMode ? " active" : ""}`,
      onclick: () => {
        state.lassoMode = !state.lassoMode;
        if (state.lassoMode) state.panMode = false;
        renderTabs();
        renderCharts();
      },
      title: "Draw freeform region",
      "aria-label": "Draw freeform region",
    }, [
      el("img", {
        class: "lasso-icon",
        src: "./assets/lasso-select.svg",
        alt: "",
        width: "44",
        height: "44",
        style: "width:44px;height:44px;",
      }),
    ]),
    renderLassoRegionActions(),
  ]);
}

function renderLassoRegionActions() {
  const hasRegions = state.lassoRegions.length > 0;
  const disabledAttrs = hasRegions ? {} : { disabled: "disabled", "aria-disabled": "true" };
  return el("div", { class: `lasso-region-actions${hasRegions ? " active" : ""}` }, [
    el("button", {
      class: "mini-btn lasso-region-btn",
      text: "Undo",
      title: "Undo last freedom region",
      onclick: undoLassoRegion,
      ...disabledAttrs,
    }),
    el("button", {
      class: "mini-btn lasso-region-btn",
      text: "Clear",
      title: "Clear all freedom regions",
      onclick: clearLassoRegions,
      ...disabledAttrs,
    }),
  ]);
}

function renderSelectionControls() {
  return el("div", { class: "selection-mode-toggle" }, [
    el("button", {
      class: `mini-btn selection-mode-btn${state.selectionMode === "single" ? " active" : ""}`,
      text: "Single point",
      title: "Keep one active point",
      onclick: () => setSelectionMode("single"),
    }),
    el("button", {
      class: `mini-btn selection-mode-btn${state.selectionMode === "multi" ? " active" : ""}`,
      text: "Multi point",
      title: "Keep up to three active points",
      onclick: () => setSelectionMode("multi"),
    }),
  ]);
}

function setSelectionMode(mode) {
  if (state.selectionMode === mode) return;
  state.selectionMode = mode;
  state.videoMode = "chunk";
  if (mode === "single" && state.selectedPoints.length > 1) {
    setSelectedPoints([state.selectedPoints[state.selectedPoints.length - 1]]);
  }
  renderTabs();
  renderCharts();
  renderPanel();
}

function renderCameraControls() {
  const cams = [
    ["left_view", "Left"],
    ["right_view", "Right"],
    ["wrist_view", "Wrist"],
  ];
  return el("div", { class: "camera-mode-toggle" }, cams.map(([cam, label]) =>
    el("button", {
      class: `mini-btn camera-mode-btn${state.cam === cam ? " active" : ""}`,
      text: label,
      title: cam.replace("robot0_", ""),
      onclick: () => setCamera(cam),
    })
  ));
}

function setCamera(cam) {
  if (state.cam === cam) return;
  state.cam = cam;
  renderTabs();
  renderPanel();
}

function renderColorControls() {
  const controls = el("div", { class: "color-controls" }, [
    el("div", { class: "color-mode-toggle" }, [
      el("button", {
        class: `mini-btn color-mode-btn${state.colorMode === "task" ? " active" : ""}`,
        text: "Task",
        title: "Color points by task",
        onclick: () => setColorMode("task"),
      }),
      el("button", {
        class: `mini-btn color-mode-btn${state.colorMode === "timestep" ? " active" : ""}`,
        text: "Timestep",
        title: "Color points by timestep",
        onclick: () => setColorMode("timestep"),
      }),
    ]),
  ]);
  controls.appendChild(renderTimestepLegend());
  return controls;
}

function renderTimestepLegend() {
  return el("div", { class: "timestep-legend", title: "Timestep color scale" }, [
    el("div", { class: "timestep-legend-title", text: "Episode timestep" }),
    el("div", {
      class: "timestep-range-control",
      onpointerdown: (event) => startTimestepRangeDrag("nearest", event),
    }, [
      el("div", { class: "timestep-ramp" }),
      el("div", {
        class: "timestep-window",
        style: `left:${state.timestepRange[0] * 100}%;right:${(1 - state.timestepRange[1]) * 100}%;`,
      }),
      el("button", {
        class: "timestep-handle timestep-handle-start",
        style: `left:${state.timestepRange[0] * 100}%;`,
        title: "Start timestep",
        "aria-label": "Start timestep",
        onpointerdown: (event) => startTimestepRangeDrag("start", event),
      }),
      el("button", {
        class: "timestep-handle timestep-handle-end",
        style: `left:${state.timestepRange[1] * 100}%;`,
        title: "End timestep",
        "aria-label": "End timestep",
        onpointerdown: (event) => startTimestepRangeDrag("end", event),
      }),
    ]),
    el("div", { class: "timestep-ticks" }, [
      el("span", { text: "start" }),
      el("span", { text: "mid" }),
      el("span", { text: "end" }),
    ]),
    el("div", {
      class: "timestep-range-label",
      text: `${Math.round(state.timestepRange[0] * 100)}-${Math.round(state.timestepRange[1] * 100)}%`,
    }),
  ]);
}

function setColorMode(mode) {
  if (state.colorMode === mode) return;
  state.colorMode = mode;
  renderTabs();
  renderCharts();
}

function startTimestepRangeDrag(handle, event) {
  event.preventDefault();
  event.stopPropagation();
  const track = event.currentTarget.closest(".timestep-range-control") || event.currentTarget;
  const rect = track.getBoundingClientRect();
  const chooseHandle = (clientX) => {
    if (handle !== "nearest") return handle;
    const value = clamp((clientX - rect.left) / rect.width, 0, 1);
    return Math.abs(value - state.timestepRange[0]) <= Math.abs(value - state.timestepRange[1])
      ? "start"
      : "end";
  };
  const activeHandle = chooseHandle(event.clientX);
  const updateFromClientX = (clientX) => {
    const value = clamp((clientX - rect.left) / rect.width, 0, 1);
    if (activeHandle === "start") setTimestepRange(value, state.timestepRange[1]);
    else setTimestepRange(state.timestepRange[0], value);
  };
  const onMove = (moveEvent) => updateFromClientX(moveEvent.clientX);
  const onUp = () => {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
  };
  updateFromClientX(event.clientX);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onUp);
}

function setTimestepRange(start, end) {
  const minGap = 0.01;
  let nextStart = clamp(start, 0, 1);
  let nextEnd = clamp(end, 0, 1);
  if (nextStart > nextEnd - minGap) {
    if (start !== state.timestepRange[0]) nextStart = Math.max(0, nextEnd - minGap);
    else nextEnd = Math.min(1, nextStart + minGap);
  }
  state.timestepRange = [nextStart, nextEnd];
  ensureSelectedPoint(true);
  renderTabs();
  renderCharts();
  renderPanel();
}

function renderSpacingControls() {
  return el("div", { class: "spacing-controls" }, [
    el("div", { class: "spacing-mode-toggle" }, [
      el("button", {
        class: `mini-btn spacing-mode-btn${state.spacingMode === "original" ? " active" : ""}`,
        text: "Original",
        title: "Use original t-SNE distances",
        onclick: () => setSpacingMode("original"),
      }),
      el("button", {
        class: `mini-btn spacing-mode-btn${state.spacingMode === "soft" ? " active" : ""}`,
        text: "Soft",
        title: "Soft-normalize consecutive episode step distances",
        onclick: () => setSpacingMode("soft"),
      }),
      el("button", {
        class: `mini-btn spacing-mode-btn${state.spacingMode === "equal" ? " active" : ""}`,
        text: "Equal",
        title: "Make consecutive episode step distances equal",
        onclick: () => setSpacingMode("equal"),
      }),
    ]),
  ]);
}

function setSpacingMode(mode) {
  if (state.spacingMode === mode) return;
  state.spacingMode = mode;
  resetSelectedChartViews();
  renderTabs();
  renderCharts();
}

function resetSelectedChartViews() {
  for (const chart of state.selectedCharts) {
    const chartState = getChartState(chartKey(chart.runId, chart.feature));
    chartState.zoom = 1;
    chartState.center = null;
  }
}

async function toggleFeature(feature) {
  const runId = state.activeRunId;
  if (!runId) return;
  const isSelected = hasChart(runId, feature);
  if (isSelected) {
    state.selectedCharts = state.selectedCharts.filter((chart) =>
      !(chart.runId === runId && chart.feature === feature)
    );
  } else {
    await loadChartData(runId, feature);
    state.selectedCharts.push({ runId, feature });
  }
  state.preferredFeatures = getSelectedFeaturesForRun(runId);
  ensureSelectedPoint(true);
}

function hasChart(runId, feature) {
  return state.selectedCharts.some((chart) => chart.runId === runId && chart.feature === feature);
}

function getChartState(key) {
  if (!state.chartStates.has(key)) {
    state.chartStates.set(key, {
      zoom: 1,
      center: null,
      baseView: null,
    });
  }
  return state.chartStates.get(key);
}

function getZoomReadout() {
  if (!state.selectedCharts.length) return "1.00x";
  const values = state.selectedCharts.map((chart) => getChartState(chartKey(chart.runId, chart.feature)).zoom);
  const first = values[0];
  return values.every((value) => Math.abs(value - first) < 1e-9)
    ? `${first.toFixed(2)}x`
    : "multi";
}

function undoLassoRegion() {
  if (!state.lassoRegions.length) return;
  state.lassoRegions.pop();
  renderTabs();
  renderCharts();
}

function clearLassoRegions() {
  if (!state.lassoRegions.length) return;
  state.lassoRegions = [];
  renderTabs();
  renderCharts();
}

function scaleSelectedCharts(factor) {
  for (const chart of state.selectedCharts) {
    const key = chartKey(chart.runId, chart.feature);
    const chartState = getChartState(key);
    chartState.zoom = Math.min(8, Math.max(0.25, chartState.zoom * factor));
    updateChartViewBox(chart.runId, chart.feature);
  }
  const readout = document.querySelector(".zoom-readout");
  if (readout) readout.textContent = getZoomReadout();
}

const ACTION_CHUNK_STEPS = 16;

function setVideoPlaybackMode(mode) {
  if (!['chunk', 'episode'].includes(mode)) return;
  state.videoMode = mode;
  renderTabs();
  renderCharts();
  renderPanel();
}

function renderVideoStrip(stage) {
  const selections = state.selectedPoints;
  if (!selections.length) return;
  const cards = selections.flatMap((selection, index) => {
    const sequences = getSequencesForRun(selection.runId);
    const manifest = state.runManifestsById.get(selection.runId) || state.runManifest;
    const seq = sequences?.sequences?.[selection.seq];
    if (!seq || !seq.videos) return null;
    const cams = Object.keys(seq.videos);
    if (!cams.length) return null;
    const shownCams = state.selectionMode === "single"
      ? ["left_view", "right_view", "wrist_view"].filter((cam) => seq.videos[cam])
      : [seq.videos[state.cam] ? state.cam : cams[0]];
    const fps = manifest.fps || 20;
    const videoStartFrame = numericValue(seq.video_start_frame) ?? 0;
    const requestedChunkStart = Math.max(0, (selection.frame - videoStartFrame) / fps);
    const selectedRun = getRunById(selection.runId);
    const selectionAccent = getSelectionAccent(selection);
    return shownCams.map((cam) => {
      const video = el("video", {
        controls: "controls",
        muted: "muted",
        playsinline: "playsinline",
        "data-sync-video": "1",
        "data-run-id": selection.runId,
        "data-seq": String(selection.seq),
        "data-selection-key": selectionKey(selection),
        "data-selection-index": String(index),
        "data-start-frame": String(videoStartFrame),
        "data-fps": String(fps),
        "data-playback-mode": state.videoMode,
        "data-chunk-steps": String(ACTION_CHUNK_STEPS),
        "data-chunk-start": String(requestedChunkStart),
        "data-loop-duration": String(ACTION_CHUNK_STEPS / fps),
        src: `./${seq.videos[cam]}`,
        ...(state.selectionMode === "multi"
          ? { style: `border-color:${selectionAccent};box-shadow:0 0 0 1px ${selectionAccent};` }
          : {}),
      });
      video.addEventListener("loadedmetadata", () => {
        const duration = Number.isFinite(video.duration) ? video.duration : requestedChunkStart + ACTION_CHUNK_STEPS / fps;
        const lastSeekableTime = Math.max(0, duration - 0.04);
        const chunkStart = Math.min(requestedChunkStart, lastSeekableTime);
        const chunkEnd = Math.min(duration, chunkStart + ACTION_CHUNK_STEPS / fps);
        video.dataset.chunkStart = String(chunkStart);
        video.dataset.chunkEnd = String(Math.max(chunkStart, chunkEnd));
        video.dataset.ready = "1";
        video.currentTime = chunkStart;
        updateActionChunkCursorFromVideo(video);
        maybeStartSynchronizedVideos();
      });
      video.addEventListener("timeupdate", () => syncVideosFrom(video));
      video.addEventListener("seeking", () => syncVideosFrom(video, { forceTime: true }));
      video.addEventListener("play", () => syncVideosFrom(video, { syncPlayback: true }));
      video.addEventListener("pause", () => syncVideosFrom(video, { syncPlayback: true }));
      video.addEventListener("ended", () => handleEpisodePlaybackEnded(video));
      return el("div", { class: "video-card" }, [
        el("div", {
          class: "video-label",
          text: `${index + 1}. ${selectedRun?.label || selection.runId}`,
        }),
        video,
      ]);
    });
  }).filter(Boolean);
  if (!cards.length) return;

  const playbackButton = el("button", {
    class: "mini-btn video-playback-btn",
    text: state.videoMode === "chunk" ? "Play episode" : "Loop 16 steps",
    title: state.videoMode === "chunk"
      ? "Play every selected episode from its selected point"
      : "Return to the selected 16-step action chunks",
    onclick: () => setVideoPlaybackMode(state.videoMode === "chunk" ? "episode" : "chunk"),
  });
  stage.appendChild(el("div", { class: "video-playback-toolbar" }, [
    el("span", {
      class: "video-playback-status",
      text: state.videoMode === "chunk" ? "Selected action chunk: 16-step loop" : "Episode playback from selected points",
    }),
    playbackButton,
  ]));
  stage.appendChild(el("div", { class: "video-strip" }, cards));
}

function getSyncVideos() {
  return [...document.querySelectorAll("video[data-sync-video='1']")];
}

function getVideoPlaybackStart(video) {
  return numericValue(video.dataset.chunkStart) ?? 0;
}

function getVideoPlaybackEnd(video) {
  if (state.videoMode === "episode") {
    return Number.isFinite(video.duration) ? video.duration : Infinity;
  }
  return numericValue(video.dataset.chunkEnd) ?? getVideoPlaybackStart(video);
}

function maybeStartSynchronizedVideos() {
  const videos = getSyncVideos();
  if (!videos.length || videos.some((video) => video.dataset.ready !== "1")) return;
  if (videos.every((video) => video.dataset.started === "1")) return;
  state.syncingVideos = true;
  try {
    for (const video of videos) {
      video.dataset.started = "1";
      video.currentTime = getVideoPlaybackStart(video);
      video.playbackRate = 1;
    }
  } finally {
    state.syncingVideos = false;
  }
  if (state.videoMode === "chunk") {
    startChunkPlaybackClock(videos);
    return;
  }
  requestAnimationFrame(() => {
    for (const video of videos) {
      const playPromise = video.play();
      if (playPromise && typeof playPromise.catch === "function") playPromise.catch(() => {});
    }
  });
}

function handleEpisodePlaybackEnded(video) {
  if (state.videoMode !== "episode") return;
  video.dataset.episodeEnded = "1";
  const videos = getSyncVideos();
  const allEnded = videos.length > 0 && videos.every((item) =>
    item.dataset.episodeEnded === "1"
    || (Number.isFinite(item.duration) && item.currentTime >= item.duration - 0.06)
  );
  if (!allEnded) return;
  setTimeout(() => {
    if (state.videoMode === "episode") setVideoPlaybackMode("chunk");
  }, 0);
}

function startChunkPlaybackClock(videos) {
  const fps = numericValue(videos[0]?.dataset.fps) || 20;
  const loopDuration = ACTION_CHUNK_STEPS / fps;
  const generation = ++state.videoClockGeneration;
  const startedAt = performance.now();

  const tick = (now) => {
    if (generation !== state.videoClockGeneration || state.videoMode !== "chunk") return;
    if (videos.some((video) => !video.isConnected || video.dataset.ready !== "1")) return;
    const relativeTime = ((now - startedAt) / 1000) % loopDuration;
    state.syncingVideos = true;
    try {
      for (const video of videos) {
        const start = getVideoPlaybackStart(video);
        const validEnd = getVideoPlaybackEnd(video);
        const lastValidTime = Math.max(start, validEnd - 0.025);
        const withinValidFrames = start + relativeTime < validEnd - 0.015;
        const targetTime = withinValidFrames ? start + relativeTime : lastValidTime;
        if (Math.abs(video.currentTime - targetTime) > 0.065) {
          video.currentTime = targetTime;
        }
        if (withinValidFrames && video.paused) {
          const playPromise = video.play();
          if (playPromise && typeof playPromise.catch === "function") playPromise.catch(() => {});
        } else if (!withinValidFrames && !video.paused) {
          video.pause();
        }
      }
    } finally {
      state.syncingVideos = false;
    }
    updateActionChunkCursor(clamp(relativeTime * fps, 0, ACTION_CHUNK_STEPS - 1));
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function syncVideosFrom(source, options = {}) {
  if (state.syncingVideos || source.dataset.ready !== "1") return;
  if (state.videoMode === "chunk") return;
  const videos = getSyncVideos().filter((video) => video.dataset.ready === "1");
  if (!videos.length) return;
  const { syncPlayback = false, forceTime = false } = options;
  const shouldSyncPlayback = syncPlayback && !source.ended;
  const threshold = forceTime ? 0.001 : 0.05;
  const sourceStart = getVideoPlaybackStart(source);
  const relativeTime = Math.max(0, source.currentTime - sourceStart);

  state.syncingVideos = true;
  try {
    for (const video of videos) {
      const end = getVideoPlaybackEnd(video);
      const start = getVideoPlaybackStart(video);
      const lastTime = Number.isFinite(end) ? Math.max(start, end - 0.025) : start + relativeTime;
      const targetTime = Math.min(start + relativeTime, lastTime);
      if (Math.abs(video.currentTime - targetTime) > threshold) {
        video.currentTime = targetTime;
      }
      if (video.playbackRate !== source.playbackRate) {
        video.playbackRate = source.playbackRate;
      }
      if (shouldSyncPlayback) {
        if (source.paused && !video.paused) video.pause();
        if (!source.paused && video.paused) {
          const playPromise = video.play();
          if (playPromise && typeof playPromise.catch === "function") playPromise.catch(() => {});
        }
      }
    }
  } finally {
    state.syncingVideos = false;
  }
  updateActionChunkCursorFromVideo(source);
}

function renderCharts() {
  const grid = document.querySelector(".chart-grid");
  if (!grid) return;
  grid.innerHTML = "";
  grid.className = `chart-grid chart-count-${Math.min(state.selectedCharts.length, 6)}`;
  if (!state.selectedCharts.length) {
    grid.appendChild(el("div", { class: "status", text: "No feature selected." }));
    return;
  }
  for (const chart of state.selectedCharts) {
    const run = getRunById(chart.runId);
    const label = `${run?.label || chart.runId} / ${chart.feature}`;
    const card = el("section", {
      class: "chart-card",
      "data-chart": chartKey(chart.runId, chart.feature),
    }, [
      el("div", { class: "chart-card-head" }, [
        el("span", { class: "chart-card-title", text: label }),
        el("button", {
          class: "chart-off-btn",
          text: "Off",
          title: `Hide ${label}`,
          onclick: () => removeChartSelection(chart.runId, chart.feature),
        }),
      ]),
      el("div", {
        class: "chart-wrap",
        "data-run": chart.runId,
        "data-feature": chart.feature,
        "data-chart": chartKey(chart.runId, chart.feature),
      }),
    ]);
    grid.appendChild(card);
    renderChart(chart.runId, chart.feature);
  }
}

function selectPoint(runId, feature, point) {
  state.videoMode = "chunk";
  const nextSelection = buildSelection(runId, point, feature);
  if (state.selectionMode === "single") {
    setSelectedPoints([nextSelection]);
  } else {
    const key = logicalSelectionKey(nextSelection);
    if (state.selectedPoints.some((selection) => logicalSelectionKey(selection) === key)) return;
    setSelectedPoints([...state.selectedPoints.slice(-2), nextSelection]);
  }
  renderTabs();
  renderCharts();
  renderPanel();
}

function renderChart(runId, feature) {
  const key = chartKey(runId, feature);
  const wrap = document.querySelector(`.chart-wrap[data-chart="${key}"]`);
  const chartState = getChartState(key);
  const pointPayload = state.pointsByChart.get(key);
  if (!wrap || !pointPayload) return;
  const sequences = state.sequencesByRunId.get(runId);
  const allPoints = applyTemporalSpacing(pointPayload.points.map((row) => ({
    x: row[0], y: row[1], seq: row[2], anchor: row[3], frame: row[4], progress: row[5],
  })));
  const visibleKeys = getVisibleSequenceKeys();
  const visiblePoints = allPoints.filter((point) => {
    const seq = sequences.sequences[point.seq];
    return isSequenceVisibleForRun(runId, point.seq, visibleKeys)
      && isPointInTimestepRange(point, seq);
  });
  if (state.selectedPoints.length) {
    setSelectedPoints(state.selectedPoints.filter(isSelectionVisible));
  }
  if (state.selected && !isSelectionVisible(state.selected)) {
    clearSelectedPoints();
    ensureSelectedPoint(true);
  }
  if (!visiblePoints.length) {
    wrap.innerHTML = `<p class="status">No points in the selected timestep range.</p>`;
    return;
  }
  const xs = visiblePoints.map((p) => p.x);
  const ys = visiblePoints.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = (maxX - minX || 1) * 0.08;
  const padY = (maxY - minY || 1) * 0.08;
  const baseX = minX - padX;
  const baseY = minY - padY;
  const baseWidth = (maxX - minX) + 2 * padX;
  const baseHeight = (maxY - minY) + 2 * padY;
  chartState.baseView = { baseX, baseY, baseWidth, baseHeight };
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.dataset.run = runId;
  svg.dataset.feature = feature;
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("class", "chart-pan-surface");
  bg.setAttribute("x", String(baseX));
  bg.setAttribute("y", String(baseY));
  bg.setAttribute("width", String(baseWidth));
  bg.setAttribute("height", String(baseHeight));
  bg.setAttribute("fill", "transparent");
  svg.appendChild(bg);

  const bySeq = new Map();
  for (const point of visiblePoints) {
    if (!bySeq.has(point.seq)) bySeq.set(point.seq, []);
    bySeq.get(point.seq).push(point);
  }
  for (const [seqId, seqPoints] of bySeq) {
    seqPoints.sort((a, b) => a.anchor - b.anchor);
    const seq = sequences.sequences[seqId];
    const seqKey = episodeKey(runId, seqId);
    const seqSelectionKey = getEpisodeSelectionKey(runId, seqId);
    const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
    area.setAttribute("d", getSequencePathD(seqPoints));
    area.setAttribute("class", `episode-area${isFocusedEpisode(runId, seqId) ? " episode-active" : ""}`);
    area.dataset.episodeKey = seqKey;
    if (seqSelectionKey) area.dataset.episodeSelectionKey = seqSelectionKey;
    area.addEventListener("mouseenter", () => setHoveredEpisode(runId, seqId));
    area.addEventListener("mouseleave", () => setHoveredEpisode(null, null));
    area.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title"))
      .textContent = `${seq.task_name}\n${seq.description}`;
    svg.appendChild(area);
  }
  for (const [seqId, seqPoints] of bySeq) {
    const seq = sequences.sequences[seqId];
    const seqKey = episodeKey(runId, seqId);
    const seqSelectionKey = getEpisodeSelectionKey(runId, seqId);
    if (state.colorMode === "timestep") {
      for (let i = 1; i < seqPoints.length; i++) {
        const prev = seqPoints[i - 1];
        const curr = seqPoints[i];
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M${prev.x},${prev.y} L${curr.x},${curr.y}`);
        path.setAttribute("class", "path-line timestep-path-line");
        path.setAttribute("stroke", getTimestepColor(curr, seq));
        path.dataset.episodeKey = seqKey;
        if (seqSelectionKey) path.dataset.episodeSelectionKey = seqSelectionKey;
        path.addEventListener("mouseenter", () => setHoveredEpisode(runId, seqId));
        path.addEventListener("mouseleave", () => setHoveredEpisode(null, null));
        svg.appendChild(path);
      }
    } else {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", getSequencePathD(seqPoints));
      path.setAttribute("class", "path-line");
      path.setAttribute("stroke", getTaskColor(seq));
      path.dataset.episodeKey = seqKey;
      if (seqSelectionKey) path.dataset.episodeSelectionKey = seqSelectionKey;
      path.addEventListener("mouseenter", () => setHoveredEpisode(runId, seqId));
      path.addEventListener("mouseleave", () => setHoveredEpisode(null, null));
      svg.appendChild(path);
    }
  }
  for (const point of visiblePoints) {
    const globalKey = getPointGlobalKey(runId, point);
    if (!globalKey) continue;
    state.lassoRegions.forEach((region, index) => {
      if (!region.keys.has(globalKey)) return;
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      marker.setAttribute("cx", point.x);
      marker.setAttribute("cy", point.y);
      marker.setAttribute("r", String(1.85 + Math.min(index, 4) * 0.16));
      marker.setAttribute("class", "lasso-point-highlight");
      marker.setAttribute("fill", region.fill);
      marker.setAttribute("stroke", region.stroke);
      svg.appendChild(marker);
    });
  }
  for (const point of visiblePoints) {
    const chartPoint = { ...point, runId, feature };
    const seq = sequences.sequences[point.seq];
    const seqSelectionKey = getEpisodeSelectionKey(runId, point.seq);
    const pointColor = getPointColor(point, seq);
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", point.x);
    c.setAttribute("cy", point.y);
    c.setAttribute("r", "0.55");
    c.setAttribute("class", "dot");
    c.setAttribute("fill", pointColor);
    c.dataset.episodeKey = episodeKey(runId, point.seq);
    if (seqSelectionKey) c.dataset.episodeSelectionKey = seqSelectionKey;
    c.dataset.seq = String(point.seq);
    c.dataset.anchor = String(point.anchor);
    c.dataset.frame = String(point.frame);
    c.addEventListener("mouseenter", () => setHoveredEpisode(runId, point.seq));
    c.addEventListener("mouseleave", () => setHoveredEpisode(null, null));
    c.addEventListener("click", () => {
      selectPoint(runId, feature, point);
    });
    c.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title"))
      .textContent = `${seq.task_name}\n${seq.description}\nframe ${point.frame}`;
    svg.appendChild(c);
  }
  for (const point of visiblePoints) {
    const chartPoint = { ...point, runId, feature };
    if (!isSelected(chartPoint)) continue;
    const seq = sequences.sequences[point.seq];
    const seqSelectionKey = getEpisodeSelectionKey(runId, point.seq);
    const pointColor = getPointColor(point, seq);
    const selectedCenterColor = complementColor(getTaskColor(seq));
    const selectedAccent = getSelectionAccent(buildSelection(runId, point, feature));
    const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    ring.setAttribute("cx", point.x);
    ring.setAttribute("cy", point.y);
    ring.setAttribute("r", "2.1");
    ring.setAttribute("class", "selected-ring");
    ring.setAttribute("fill", pointColor);
    ring.setAttribute("stroke", selectedAccent);
    ring.dataset.episodeKey = episodeKey(runId, point.seq);
    if (seqSelectionKey) ring.dataset.episodeSelectionKey = seqSelectionKey;
    ring.dataset.run = runId;
    ring.dataset.seq = String(point.seq);
    ring.dataset.anchor = String(point.anchor);
    ring.dataset.frame = String(point.frame);
    ring.addEventListener("mouseenter", () => setHoveredEpisode(runId, point.seq));
    ring.addEventListener("mouseleave", () => setHoveredEpisode(null, null));
    ring.addEventListener("click", () => {
      selectPoint(runId, feature, point);
    });
    svg.appendChild(ring);

    const center = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    center.setAttribute("cx", point.x);
    center.setAttribute("cy", point.y);
    center.setAttribute("r", "0.72");
    center.setAttribute("class", "dot selected");
    center.setAttribute("fill", selectedCenterColor);
    center.dataset.episodeKey = episodeKey(runId, point.seq);
    if (seqSelectionKey) center.dataset.episodeSelectionKey = seqSelectionKey;
    center.dataset.run = runId;
    center.dataset.seq = String(point.seq);
    center.dataset.anchor = String(point.anchor);
    center.dataset.frame = String(point.frame);
    center.addEventListener("mouseenter", () => setHoveredEpisode(runId, point.seq));
    center.addEventListener("mouseleave", () => setHoveredEpisode(null, null));
    center.addEventListener("click", () => {
      selectPoint(runId, feature, point);
    });
    svg.appendChild(center);
  }
  wrap.innerHTML = "";
  wrap.classList.toggle("pan-enabled", state.panMode);
  wrap.classList.toggle("lasso-enabled", state.lassoMode);
  wrap.appendChild(svg);
  updateChartViewBox(runId, feature);
  attachChartLasso(svg, runId, feature, visiblePoints);
  attachChartPan(svg, runId, feature);
}

function getChartCenter(runId, feature) {
  const chartState = getChartState(chartKey(runId, feature));
  if (!chartState.baseView) return { x: 0, y: 0 };
  if (chartState.center) return chartState.center;
  return {
    x: chartState.baseView.baseX + (chartState.baseView.baseWidth / 2),
    y: chartState.baseView.baseY + (chartState.baseView.baseHeight / 2),
  };
}

function updateChartViewBox(runId, feature) {
  const key = chartKey(runId, feature);
  const svg = document.querySelector(`.chart-wrap[data-chart="${key}"] svg`);
  const chartState = getChartState(key);
  if (!svg || !chartState.baseView) return;
  const { baseWidth, baseHeight } = chartState.baseView;
  const center = getChartCenter(runId, feature);
  const zoomWidth = baseWidth / chartState.zoom;
  const zoomHeight = baseHeight / chartState.zoom;
  svg.setAttribute(
    "viewBox",
    `${center.x - (zoomWidth / 2)} ${center.y - (zoomHeight / 2)} ${zoomWidth} ${zoomHeight}`
  );
  const readout = document.querySelector(".zoom-readout");
  if (readout) readout.textContent = getZoomReadout();
}

function clientPointToSvg(svg, clientX, clientY) {
  const matrix = svg.getScreenCTM();
  if (!matrix) return null;
  const point = svg.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  const transformed = point.matrixTransform(matrix.inverse());
  return { x: transformed.x, y: transformed.y };
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const pi = polygon[i];
    const pj = polygon[j];
    const intersects = ((pi.y > point.y) !== (pj.y > point.y))
      && point.x < ((pj.x - pi.x) * (point.y - pi.y)) / ((pj.y - pi.y) || 1e-9) + pi.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function attachChartLasso(svg, runId, feature, visiblePoints) {
  svg.addEventListener("pointerdown", (event) => {
    if (!state.lassoMode || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    svg.setPointerCapture(event.pointerId);
    const regionColor = lassoPalette[state.lassoRegions.length % lassoPalette.length];
    const polygon = [];
    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    path.setAttribute("class", "lasso-draft");
    path.setAttribute("fill", regionColor.fill);
    path.setAttribute("stroke", regionColor.stroke);
    svg.appendChild(path);

    const addPoint = (pointerEvent) => {
      const point = clientPointToSvg(svg, pointerEvent.clientX, pointerEvent.clientY);
      if (!point) return;
      const last = polygon[polygon.length - 1];
      if (last && Math.hypot(point.x - last.x, point.y - last.y) < 0.4) return;
      polygon.push(point);
      path.setAttribute("points", polygon.map((p) => `${p.x},${p.y}`).join(" "));
    };

    const onMove = (moveEvent) => addPoint(moveEvent);
    const onUp = () => {
      svg.releasePointerCapture(event.pointerId);
      svg.removeEventListener("pointermove", onMove);
      svg.removeEventListener("pointerup", onUp);
      svg.removeEventListener("pointercancel", onUp);
      if (path.parentNode) path.parentNode.removeChild(path);
      if (polygon.length >= 3) {
        const nextKeys = new Set();
        for (const point of visiblePoints) {
          if (!pointInPolygon(point, polygon)) continue;
          const key = getPointGlobalKey(runId, point);
          if (key) nextKeys.add(key);
        }
        if (nextKeys.size) {
          state.lassoRegions.push({
            keys: nextKeys,
            fill: regionColor.fill,
            stroke: regionColor.stroke,
          });
          renderTabs();
          renderCharts();
        }
      }
    };

    addPoint(event);
    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerup", onUp);
    svg.addEventListener("pointercancel", onUp);
  });
}

function zoomChartAt(runId, feature, clientX, clientY, direction) {
  const key = chartKey(runId, feature);
  const svg = document.querySelector(`.chart-wrap[data-chart="${key}"] svg`);
  const chartState = getChartState(key);
  if (!svg || !chartState.baseView) return;
  const bounds = svg.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;

  const factor = direction < 0 ? 1.2 : (1 / 1.2);
  const nextZoom = Math.min(8, Math.max(0.25, chartState.zoom * factor));
  if (nextZoom === chartState.zoom) return;

  const currentCenter = getChartCenter(runId, feature);
  const currentWidth = chartState.baseView.baseWidth / chartState.zoom;
  const currentHeight = chartState.baseView.baseHeight / chartState.zoom;
  const currentLeft = currentCenter.x - (currentWidth / 2);
  const currentTop = currentCenter.y - (currentHeight / 2);
  const fracX = (clientX - bounds.left) / bounds.width;
  const fracY = (clientY - bounds.top) / bounds.height;
  const worldX = currentLeft + (fracX * currentWidth);
  const worldY = currentTop + (fracY * currentHeight);

  chartState.zoom = nextZoom;
  const nextWidth = chartState.baseView.baseWidth / chartState.zoom;
  const nextHeight = chartState.baseView.baseHeight / chartState.zoom;
  chartState.center = {
    x: worldX - (fracX * nextWidth) + (nextWidth / 2),
    y: worldY - (fracY * nextHeight) + (nextHeight / 2),
  };
  updateChartViewBox(runId, feature);
}

function attachChartPan(svg, runId, feature) {
  const surface = svg.querySelector(".chart-pan-surface");
  if (!surface) return;
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomChartAt(runId, feature, event.clientX, event.clientY, event.deltaY);
  }, { passive: false });
  svg.addEventListener("pointerdown", (event) => {
    const allowLeftPan = event.button === 0 && state.panMode;
    const allowMiddlePan = event.button === 1;
    const chartState = getChartState(chartKey(runId, feature));
    if (!chartState.baseView || (!allowLeftPan && !allowMiddlePan)) return;
    event.preventDefault();
    svg.setPointerCapture(event.pointerId);
    svg.classList.add("panning");
    const bounds = svg.getBoundingClientRect();
    const startCenter = getChartCenter(runId, feature);
    const zoomWidth = chartState.baseView.baseWidth / chartState.zoom;
    const zoomHeight = chartState.baseView.baseHeight / chartState.zoom;

    const onMove = (moveEvent) => {
      const dx = ((moveEvent.clientX - event.clientX) / bounds.width) * zoomWidth;
      const dy = ((moveEvent.clientY - event.clientY) / bounds.height) * zoomHeight;
      chartState.center = { x: startCenter.x - dx, y: startCenter.y - dy };
      updateChartViewBox(runId, feature);
    };
    const onUp = () => {
      svg.classList.remove("panning");
      svg.removeEventListener("pointermove", onMove);
      svg.removeEventListener("pointerup", onUp);
      svg.removeEventListener("pointercancel", onUp);
    };

    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerup", onUp);
    svg.addEventListener("pointercancel", onUp);
  });
}

function isSelected(point) {
  const key = getPointSelectionKey(point.runId, point);
  return Boolean(key && state.selectedPoints.some((selection) => logicalSelectionKey(selection) === key));
}

function getSelectedEpisodeKey() {
  return state.selected ? episodeKey(state.selected.runId, state.selected.seq) : null;
}

function getSelectedEpisodeSelectionKey() {
  return state.selected?.episodeSelectionKey || null;
}

function isFocusedEpisode(runId, seqId) {
  const key = episodeKey(runId, seqId);
  const episodeSelectionKey = getEpisodeSelectionKey(runId, seqId);
  return key === state.hoveredEpisodeKey
    || key === getSelectedEpisodeKey()
    || Boolean(episodeSelectionKey && episodeSelectionKey === getSelectedEpisodeSelectionKey());
}

function updateEpisodeFocus() {
  const selectedKey = getSelectedEpisodeKey();
  const selectedEpisodeSelectionKey = getSelectedEpisodeSelectionKey();
  document.querySelectorAll("[data-episode-key]").forEach((node) => {
    const key = node.dataset.episodeKey;
    const episodeSelectionKey = node.dataset.episodeSelectionKey;
    const selectedByManifest = Boolean(
      selectedEpisodeSelectionKey && episodeSelectionKey === selectedEpisodeSelectionKey
    );
    node.classList.toggle("episode-hovered", Boolean(state.hoveredEpisodeKey && key === state.hoveredEpisodeKey));
    node.classList.toggle("episode-active", Boolean((selectedKey && key === selectedKey) || selectedByManifest));
    node.classList.remove("episode-dimmed");
  });
}

function setHoveredEpisode(runId, seqId) {
  state.hoveredEpisodeKey = runId === null ? null : episodeKey(runId, seqId);
  updateEpisodeFocus();
}

function getSequencePathD(seqPoints) {
  return seqPoints.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
}

function getVideoSelectionIndex(video) {
  const index = Number(video.dataset.selectionIndex);
  if (Number.isInteger(index) && index >= 0 && index < state.selectedPoints.length) return index;
  return state.selectedPoints.findIndex((point) => selectionKey(point) === video.dataset.selectionKey);
}

function updateActionChunkCursorFromVideo(video) {
  if (!video) return;
  const selectionIndex = getVideoSelectionIndex(video);
  const selection = selectionIndex >= 0 ? state.selectedPoints[selectionIndex] : null;
  if (!selection) return;
  if (video.dataset.runId !== selection.runId) return;
  if (Number(video.dataset.seq) !== Number(selection.seq)) return;
  const fps = numericValue(video.dataset.fps) || 20;
  const chunk = getActionChunk(selection);
  const horizon = Math.max(1, chunk?.values?.length || ACTION_CHUNK_STEPS);
  const chunkStart = numericValue(video.dataset.chunkStart) ?? 0;
  const timestep = clamp((video.currentTime - chunkStart) * fps, 0, Math.max(0, horizon - 1));
  updateActionChunkCursor(timestep);
}

function updateActionChunkCursor(timestep) {
  const svg = document.querySelector(".action-chunk-chart");
  if (!svg) return;
  const playhead = svg.querySelector(".action-playhead[data-action-playhead='1']");
  const band = svg.querySelector(".action-playhead-band[data-action-playhead-band='1']");
  if (!playhead || !band) return;
  const left = numericValue(svg.dataset.plotLeft) ?? 0;
  const width = numericValue(svg.dataset.plotWidth) ?? 0;
  const horizon = Math.max(1, (numericValue(svg.dataset.horizon) ?? 1) - 1);
  const x = left + (clamp(timestep, 0, horizon) / horizon) * width;
  playhead.setAttribute("x1", x.toFixed(2));
  playhead.setAttribute("x2", x.toFixed(2));
  band.setAttribute("x", (x - 2.5).toFixed(2));
}

function preserveTaskPanelScroll() {
  const panel = document.querySelector(".panel");
  state.pendingPanelScrollTop = panel ? panel.scrollTop : 0;
  state.pendingTaskPanelScrollTop = panel?.querySelector(".task-panel")?.scrollTop || 0;
}

function renderPanel() {
  const panel = document.querySelector(".panel");
  const savedScrollTop = state.pendingPanelScrollTop ?? (panel ? panel.scrollTop : 0);
  const savedTaskPanelScrollTop =
    state.pendingTaskPanelScrollTop ?? (panel?.querySelector(".task-panel")?.scrollTop || 0);
  state.pendingPanelScrollTop = null;
  state.pendingTaskPanelScrollTop = null;
  const selectedSequences = state.selected
    ? (state.sequencesByRunId.get(state.selected.runId) || state.sequences)
    : null;
  const seq = state.selected ? selectedSequences.sequences[state.selected.seq] : null;
  panel.innerHTML = "";
  renderTaskDescriptionPanel(panel);
  panel.appendChild(el("h2", { text: "Selection" }));
  if (!state.selectedPoints.length || !seq) {
    panel.appendChild(el("p", { class: "status", text: "Click a trajectory point." }));
    return;
  }
  if (state.selectedPoints.length > 1) {
    panel.appendChild(el("div", { class: "selection-list" }, state.selectedPoints.map((selection, index) => {
      const selectionSeqs = state.sequencesByRunId.get(selection.runId) || state.sequences;
      const selectionSeq = selectionSeqs?.sequences?.[selection.seq];
      const selectionRun = getRunById(selection.runId);
      return el("div", { class: `selection-chip${selection === state.selected ? " active" : ""}` }, [
        el("strong", { text: `${index + 1}. ${selectionRun?.label || selection.runId}` }),
        el("span", { text: `${selectionSeq?.task_name || ""} / frame ${selection.frame}` }),
      ]);
    })));
  }
  const selectedRun = getRunById(state.selected.runId);
  panel.appendChild(el("div", { class: "info" }, [
    definitionList([
      ["Run", selectedRun?.label || state.selected.runId],
      ["Feature", state.selected.feature || ""],
      ["Task", seq.task_name],
      ["Description", seq.description],
      ["Episode", String(seq.episode_index)],
      ["Frame", String(state.selected.frame), "selection-frame"],
    ]),
  ]));
  panel.appendChild(renderActionChunkPanel());
  if (!Object.keys(seq.videos || {}).length) {
    panel.appendChild(el("p", { class: "status", text: "No video available for this sequence." }));
  }
  requestAnimationFrame(() => {
    panel.scrollTop = savedScrollTop;
    const taskPanel = panel.querySelector(".task-panel");
    if (taskPanel) taskPanel.scrollTop = savedTaskPanelScrollTop;
  });
}

function renderActionChunkPanel() {
  const body = el("div", { class: "action-chunk-body" });
  const section = el("section", { class: "action-chunk-panel" }, [
    el("h3", { text: "Transformed Action Chunk" }),
    body,
  ]);
  if (!state.actionChunks && !state.actionChunksLoading && !state.actionChunksError) {
    loadActionChunks().then(renderPanel);
  }
  if (state.actionChunksLoading) {
    body.appendChild(el("p", { class: "status", text: "Loading action chunk..." }));
    return section;
  }
  if (state.actionChunksError) {
    body.appendChild(el("p", { class: "status", text: "Action chunk data is not available." }));
    return section;
  }
  const chunks = state.selectedPoints
    .map((selection, index) => ({ selection, index, chunk: getActionChunk(selection) }))
    .filter((item) => item.chunk?.values?.length);
  if (!chunks.length) {
    body.appendChild(el("p", { class: "status", text: "No action chunk for this point." }));
    return section;
  }
  body.appendChild(renderActionChunkChart(chunks));
  body.appendChild(renderActionDimLegend(chunks));
  body.appendChild(el("p", {
    class: "action-chunk-meta",
    text: `T=${chunks[0].chunk.values.length}, dim=${chunks[0].chunk.values[0]?.length || 0}`,
  }));
  return section;
}

function renderActionChunkChart(chunks) {
  const width = 320;
  const height = 170;
  const pad = { left: 34, right: 10, top: 12, bottom: 24 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const dimCount = chunks[0].chunk.values[0]?.length || 0;
  const visibleDims = visibleActionDims(dimCount);
  const flat = chunks.flatMap(({ chunk }) => (
    chunk.values.flatMap((row) => row.filter((_, dim) => visibleDims.has(dim)))
  )).map(Number).filter((value) => Number.isFinite(value));
  let minValue = Math.min(...flat);
  let maxValue = Math.max(...flat);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    minValue = -1;
    maxValue = 1;
  }
  if (Math.abs(maxValue - minValue) < 1e-9) {
    minValue -= 1;
    maxValue += 1;
  }
  const horizon = Math.max(1, chunks[0].chunk.values.length - 1);
  const xFor = (t) => pad.left + (t / horizon) * plotWidth;
  const yFor = (value) => pad.top + (1 - (value - minValue) / (maxValue - minValue)) * plotHeight;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "action-chunk-chart");
  svg.dataset.plotLeft = String(pad.left);
  svg.dataset.plotWidth = String(plotWidth);
  svg.dataset.horizon = String(chunks[0].chunk.values.length);
  svg.appendChild(svgLine(pad.left, pad.top, pad.left, height - pad.bottom, "action-axis"));
  svg.appendChild(svgLine(pad.left, height - pad.bottom, width - pad.right, height - pad.bottom, "action-axis"));
  svg.appendChild(svgText(4, pad.top + 4, maxValue.toFixed(2), "action-axis-label"));
  svg.appendChild(svgText(4, height - pad.bottom, minValue.toFixed(2), "action-axis-label"));
  const labels = actionDimLabels(dimCount);
  const dashStyles = ["", "5 3", "1.5 3", "8 3 2 3"];
  chunks.forEach(({ selection, index, chunk }) => {
    for (let dim = 0; dim < dimCount; dim++) {
      if (!visibleDims.has(dim)) continue;
      const points = chunk.values
        .map((row, t) => `${xFor(t).toFixed(2)},${yFor(Number(row[dim])).toFixed(2)}`)
        .join(" ");
      const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      polyline.setAttribute("points", points);
      polyline.setAttribute("class", "action-line");
      polyline.setAttribute("stroke", colors[dim % colors.length]);
      polyline.setAttribute("stroke-dasharray", dashStyles[index % dashStyles.length]);
      polyline.setAttribute("opacity", String(Math.max(0.46, 0.9 - index * 0.12)));
      polyline.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title"))
        .textContent = `${index + 1}. ${labels[dim]}`;
      svg.appendChild(polyline);
    }
  });
  const band = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  band.setAttribute("x", String(pad.left - 2.5));
  band.setAttribute("y", String(pad.top));
  band.setAttribute("width", "5");
  band.setAttribute("height", String(plotHeight));
  band.setAttribute("class", "action-playhead-band");
  band.setAttribute("data-action-playhead-band", "1");
  svg.appendChild(band);
  const playhead = svgLine(pad.left, pad.top, pad.left, height - pad.bottom, "action-playhead");
  playhead.setAttribute("data-action-playhead", "1");
  svg.appendChild(playhead);
  return svg;
}

function renderActionDimLegend(chunks) {
  const dimCount = chunks[0].chunk.values[0]?.length || 0;
  const labels = actionDimLabels(dimCount);
  const visibleDims = visibleActionDims(dimCount);
  const buttons = labels.map((label, dim) => el("button", {
    class: `action-dim-toggle${visibleDims.has(dim) ? " active" : ""}`,
    title: label,
    onclick: () => toggleActionDim(dim, dimCount),
  }, [
    el("span", { class: "action-dim-swatch", style: `background:${colors[dim % colors.length]}` }),
    el("span", { text: label }),
  ]));
  const dashItems = chunks.slice(0, 4).map(({ selection, index }) => {
    const seq = getSequenceById(selection.runId, selection.seq);
    return el("span", { class: "action-selection-style" }, [
      el("span", { class: `action-style-line style-${index % 4}` }),
      el("span", { text: `${index + 1}. ${seq?.task_name || ""}` }),
    ]);
  });
  return el("div", { class: "action-legend" }, [
    el("div", { class: "action-dim-toggles" }, buttons),
    ...(chunks.length > 1 ? [el("div", { class: "action-selection-legend" }, dashItems)] : []),
  ]);
}

function svgLine(x1, y1, x2, y2, className) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(x1));
  line.setAttribute("y1", String(y1));
  line.setAttribute("x2", String(x2));
  line.setAttribute("y2", String(y2));
  line.setAttribute("class", className);
  return line;
}

function svgText(x, y, text, className) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("class", className);
  node.textContent = text;
  return node;
}

function renderTaskDescriptionPanel(panel) {
  const sequences = state.sequences ? state.sequences.sequences : [];
  const byGroup = new Map();
  for (const seq of sequences) {
    const meta = getTaskMeta(seq);
    if (!byGroup.has(meta.group)) byGroup.set(meta.group, new Map());
    const byTask = byGroup.get(meta.group);
    if (!byTask.has(seq.task_name)) byTask.set(seq.task_name, []);
    byTask.get(seq.task_name).push(seq);
  }

  const allOn = el("button", {
    class: "mini-btn",
    text: "All ON",
    onmousedown: (event) => event.preventDefault(),
    onclick: () => {
      preserveTaskPanelScroll();
      state.visibleSeqs = new Set(sequences.map((seq) => seq.seq_id));
      ensureSelectedPoint(true);
      renderViewer();
    },
  });
  const allOff = el("button", {
    class: "mini-btn",
    text: "All OFF",
    onmousedown: (event) => event.preventDefault(),
    onclick: () => {
      preserveTaskPanelScroll();
      state.visibleSeqs = new Set();
      ensureSelectedPoint(true);
      renderViewer();
    },
  });

  panel.appendChild(el("h2", { text: "Task / Description / Episode" }));
  panel.appendChild(el("div", { class: "task-toolbar" }, [allOn, allOff]));

  const taskPanel = el("div", { class: "task-panel" });
  const groupNames = [...byGroup.keys()].sort((a, b) => {
    const ai = kitchenTaskGroups.indexOf(a);
    const bi = kitchenTaskGroups.indexOf(b);
    return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi) || a.localeCompare(b);
  });

  for (const groupName of groupNames) {
    const byTask = byGroup.get(groupName);
    const groupSeqs = [...byTask.values()].flat();
    const groupVisibleCount = groupSeqs.filter((seq) => state.visibleSeqs.has(seq.seq_id)).length;
    const isGroupOpen = state.openTaskGroups.has(groupName);
    const groupButton = el("button", {
      class: `task-group-name${groupVisibleCount ? " active" : " inactive"}`,
      onmousedown: (event) => event.preventDefault(),
      onclick: () => {
        preserveTaskPanelScroll();
        if (state.openTaskGroups.has(groupName)) state.openTaskGroups.delete(groupName);
        else state.openTaskGroups.add(groupName);
        renderPanel();
      },
    }, [
      el("span", { class: "task-caret", text: isGroupOpen ? "-" : "+" }),
      el("span", { text: groupName }),
      el("span", { class: "task-count", text: `${groupVisibleCount}/${groupSeqs.length}` }),
    ]);
    const groupToggle = el("button", {
      class: "task-eye",
      text: "toggle",
      title: `Toggle all ${groupName} episodes`,
      onmousedown: (event) => event.preventDefault(),
      onclick: (event) => {
        event.stopPropagation();
        preserveTaskPanelScroll();
        const nextOn = groupVisibleCount !== groupSeqs.length;
        for (const seq of groupSeqs) {
          if (nextOn) state.visibleSeqs.add(seq.seq_id);
          else state.visibleSeqs.delete(seq.seq_id);
        }
        ensureSelectedPoint(true);
        renderViewer();
      },
    });

    const taskList = el("div", { class: `task-group-list${isGroupOpen ? " open" : ""}` });
    const taskEntries = [...byTask.entries()].sort((a, b) => {
      const ai = kitchenTaskOrder.indexOf(a[0]);
      const bi = kitchenTaskOrder.indexOf(b[0]);
      return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi) || a[0].localeCompare(b[0]);
    });

    for (const [taskName, taskSeqs] of taskEntries) {
      taskSeqs.sort((a, b) => a.episode_index - b.episode_index || a.seq_id - b.seq_id);
      const taskMeta = getTaskMeta(taskSeqs[0] || taskName);
      const visibleCount = taskSeqs.filter((seq) => state.visibleSeqs.has(seq.seq_id)).length;
      const isOpen = state.openTasks.has(taskName);
      const taskButton = el("button", {
        class: `task-name${visibleCount ? " active" : " inactive"}`,
        onmousedown: (event) => event.preventDefault(),
        onclick: () => {
          preserveTaskPanelScroll();
          if (state.openTasks.has(taskName)) state.openTasks.delete(taskName);
          else state.openTasks.add(taskName);
          renderPanel();
        },
      }, [
        el("span", { class: "task-caret", text: isOpen ? "-" : "+" }),
        el("span", { text: taskMeta.label }),
        el("span", { class: "task-count", text: `${visibleCount}/${taskSeqs.length}` }),
      ]);
      const taskToggle = el("button", {
        class: "task-eye",
        text: "toggle",
        title: `Toggle all ${taskMeta.label} episodes`,
        onmousedown: (event) => event.preventDefault(),
        onclick: (event) => {
          event.stopPropagation();
          preserveTaskPanelScroll();
          const nextOn = visibleCount !== taskSeqs.length;
          for (const seq of taskSeqs) {
            if (nextOn) state.visibleSeqs.add(seq.seq_id);
            else state.visibleSeqs.delete(seq.seq_id);
          }
          ensureSelectedPoint(true);
          renderViewer();
        },
      });

      const details = el("div", { class: `desc-list${isOpen ? " open" : ""}` });
      const taskEpisodeNumber = new Map(taskSeqs.map((seq, index) => [seq.seq_id, index + 1]));
      const byDescription = new Map();
      for (const seq of taskSeqs) {
        const description = seq.description || "(no description)";
        if (!byDescription.has(description)) byDescription.set(description, []);
        byDescription.get(description).push(seq);
      }
      const descriptionEntries = [...byDescription.entries()]
        .sort((a, b) => a[0].localeCompare(b[0]));

      for (const [description, descSeqs] of descriptionEntries) {
        const descVisibleCount = descSeqs.filter((seq) => state.visibleSeqs.has(seq.seq_id)).length;
        const descriptionButton = el("button", {
          class: `desc-toggle${descVisibleCount ? " active" : " inactive"}${descVisibleCount && descVisibleCount < descSeqs.length ? " partial" : ""}`,
          text: description,
          title: `Toggle all ${descSeqs.length} matching episodes`,
          onmousedown: (event) => event.preventDefault(),
          onclick: () => {
            preserveTaskPanelScroll();
            const nextOn = descVisibleCount !== descSeqs.length;
            for (const seq of descSeqs) {
              if (nextOn) state.visibleSeqs.add(seq.seq_id);
              else state.visibleSeqs.delete(seq.seq_id);
            }
            ensureSelectedPoint(true);
            renderViewer();
          },
        });
        const episodeButtons = descSeqs.map((seq) => {
          const episodeNumber = taskEpisodeNumber.get(seq.seq_id);
          const visible = state.visibleSeqs.has(seq.seq_id);
          return el("button", {
            class: `desc-episode-toggle${visible ? " active" : " inactive"}`,
            text: String(episodeNumber),
            title: `Episode ${episodeNumber} (dataset episode ${seq.episode_index})`,
            "aria-label": `${taskMeta.label}, ${description}, episode ${episodeNumber}, ${visible ? "on" : "off"}`,
            onmousedown: (event) => event.preventDefault(),
            onclick: (event) => {
              event.stopPropagation();
              preserveTaskPanelScroll();
              if (state.visibleSeqs.has(seq.seq_id)) state.visibleSeqs.delete(seq.seq_id);
              else state.visibleSeqs.add(seq.seq_id);
              ensureSelectedPoint(true);
              renderViewer();
            },
          });
        });
        details.appendChild(el("div", { class: "desc-group" }, [
          descriptionButton,
          el("div", { class: "episode-filter-label", text: "Episodes" }),
          el("div", { class: "desc-episodes" }, episodeButtons),
        ]));
      }
      taskList.appendChild(el("div", { class: "task-accordion" }, [
        el("div", { class: "task-row" }, [taskButton, taskToggle]),
        details,
      ]));
    }

    taskPanel.appendChild(el("div", { class: "task-group" }, [
      el("div", { class: "task-group-row" }, [groupButton, groupToggle]),
      taskList,
    ]));
  }
  panel.appendChild(taskPanel);
}

function definitionList(rows) {
  const dl = el("dl");
  for (const [k, v, id] of rows) {
    const ddAttrs = id ? { text: v, id } : { text: v };
    dl.appendChild(el("div", {}, [
      el("dt", { text: k }),
      el("dd", ddAttrs),
    ]));
  }
  return dl;
}

init();
