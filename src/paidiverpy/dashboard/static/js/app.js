// -----------------------------------------------------------------------------
// Application state and shared helpers
// -----------------------------------------------------------------------------

const state = {
  registry: null,
  config: {
    general: {
      sample_data: "benthic_csv",
      output_path: "output",
      track_changes: true,
    },
    steps: [],
  },
  runId: null,
  previewManifest: null,
  yamlLocations: [],
  memorySamples: [],
  memorySteps: [],
};
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (url, options = {}) => {
  const r = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok)
    throw new Error(
      typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body),
    );
  return body;
};
const toast = (msg) => {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3000);
};
const label = (s) => s.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
const escapeHtml = (value) => {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
};
// -----------------------------------------------------------------------------
// Schema-driven form controls
// -----------------------------------------------------------------------------

function coerce(input, type) {
  if (type === "boolean") return input.checked;
  if (type === "integer") return input.value === "" ? null : Number.parseInt(input.value, 10);
  if (type === "number") return input.value === "" ? null : Number(input.value);
  if (type === "json") {
    if (!input.value.trim()) return null;
    try {
      return JSON.parse(input.value);
    } catch {
      return input.value;
    }
  }
  return input.value || null;
}
function field(def, value, onchange, path) {
  const wrap = document.createElement("div");
  wrap.className = "field" + (def.type === "json" ? " wide" : "");
  wrap.dataset.configPath = path;
  const lab = document.createElement("label");
  lab.textContent = label(def.name);
  wrap.append(lab);
  let input;
  if (def.type === "select") {
    input = document.createElement("select");
    if (def.nullable) {
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = def.name === "sample_data" ? "No sample data" : "None";
      input.append(empty);
    }
    for (const option of def.options) {
      const o = document.createElement("option");
      o.value = option;
      o.textContent = label(String(option));
      input.append(o);
    }
    input.value = value ?? "";
  } else {
    input = document.createElement("input");
    input.type =
      def.type === "boolean"
        ? "checkbox"
        : def.type === "integer" || def.type === "number"
          ? "number"
          : "text";
    if (def.type === "number") input.step = "any";
    if (def.type === "json") input.value = value == null ? "" : JSON.stringify(value);
    else if (def.type === "boolean") input.checked = Boolean(value);
    else input.value = value ?? "";
  }
  input.addEventListener("change", () => onchange(coerce(input, def.type)));
  input.addEventListener("focus", () => selectYamlPath(path));
  wrap.append(input);
  if (def.description) {
    const small = document.createElement("small");
    small.textContent = def.description;
    wrap.append(small);
  }
  return wrap;
}
// -----------------------------------------------------------------------------
// General settings and loading-time operations
// -----------------------------------------------------------------------------

function renderGeneral() {
  const root = $("#general-fields");
  root.innerHTML = "";
  for (const def of state.registry.general) {
    root.append(
      field(
        def,
        state.config.general[def.name] ?? def.default,
        (v) => {
          if (v == null || v === "") delete state.config.general[def.name];
          else state.config.general[def.name] = v;
          changed();
        },
        `general.${def.name}`,
      ),
    );
  }
}
function modeDef(category, mode) {
  return state.registry.categories
    .find((c) => c.name === category)
    ?.modes.find((m) => m.name === mode);
}
function defaultParams(def) {
  const params = {};
  for (const item of def?.fields || []) if (item.default != null) params[item.name] = item.default;
  return params;
}
function renderGeneralSteps() {
  for (const category of ["sampling", "convert"]) {
    const root = $(`#general-${category}`),
      items = state.config.general[category] || [];
    root.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "initial-empty";
      empty.textContent = `No initial ${category} configured.`;
      root.append(empty);
      continue;
    }
    items.forEach((data, index) => {
      const base = `general.${category}[${index}]`,
        def = modeDef(category, data.mode),
        card = document.createElement("article");
      card.className = "initial-item";
      card.dataset.configPath = base;
      card.style.setProperty("--step-color", colors[category]);
      card.innerHTML = `<div class="initial-item-head"><b>${index + 1}. ${def?.label || label(data.mode)}</b><button class="icon-button remove" title="Remove">×</button></div><div class="form-grid"></div>`;
      const grid = $(".form-grid", card);
      const modeField = {
        name: "mode",
        type: "select",
        options: state.registry.general_steps
          .find((item) => item.name === category)
          .modes.map((item) => item.name),
        description: `${label(category)} method.`,
      };
      grid.append(
        field(
          modeField,
          data.mode,
          (value) => {
            data.mode = value;
            data.params = defaultParams(modeDef(category, value));
            changed();
          },
          `${base}.mode`,
        ),
      );
      grid.append(
        field(
          {
            name: "test",
            type: "boolean",
            description: "Run this loading operation in test mode.",
          },
          data.test ?? false,
          (value) => {
            data.test = value;
            changed(false);
          },
          `${base}.test`,
        ),
      );
      for (const item of def?.fields || [])
        grid.append(
          field(
            item,
            data.params?.[item.name] ?? item.default,
            (value) => {
              data.params ??= {};
              if (value == null || value === "") delete data.params[item.name];
              else data.params[item.name] = value;
              changed(false);
            },
            `${base}.params.${item.name}`,
          ),
        );
      $(".remove", card).onclick = () => {
        items.splice(index, 1);
        if (!items.length) delete state.config.general[category];
        changed();
      };
      root.append(card);
    });
  }
}
function setupGeneralSteps() {
  for (const category of ["sampling", "convert"]) {
    const definition = state.registry.general_steps.find((item) => item.name === category),
      select = $(`#general-${category}-mode`);
    select.innerHTML = "";
    for (const mode of definition.modes) {
      const option = document.createElement("option");
      option.value = mode.name;
      option.textContent = mode.label;
      select.append(option);
    }
    $(`#add-general-${category}`).onclick = () => {
      const def = modeDef(category, select.value);
      state.config.general[category] ??= [];
      state.config.general[category].push({
        mode: def.name,
        test: false,
        params: defaultParams(def),
      });
      changed();
    };
  }
}
// -----------------------------------------------------------------------------
// Processing step builder
// -----------------------------------------------------------------------------

const colors = {
  position: "#ef7657",
  sampling: "#8b6ab8",
  convert: "#007f95",
  colour: "#e09b32",
  custom: "#607d8b",
};
function renderSteps() {
  const root = $("#steps");
  root.innerHTML = "";
  state.config.steps.forEach((item, index) => {
    const [category, data] = Object.entries(item)[0];
    const base = `steps[${index}].${category}`;
    const mode = category === "custom" ? "custom" : data.mode;
    const def = modeDef(category, mode);
    const card = document.createElement("article");
    card.className = "step-card";
    card.draggable = true;
    card.dataset.index = index;
    card.dataset.runtimeIndex = index + 1;
    card.dataset.configPath = base;
    card.style.setProperty("--step-color", colors[category] || "#007f95");
    card.innerHTML = `<div class="step-head"><span class="step-number">${index + 1}</span><div class="step-title"><b>${data.name || def?.label || label(mode)}</b><small>${label(category)} · ${def?.label || label(mode)}</small></div><span class="step-run-status"></span><button class="icon-button up" title="Move up">↑</button><button class="icon-button down" title="Move down">↓</button><button class="icon-button remove" title="Remove">×</button></div><div class="step-fields"><div class="form-grid"></div></div>`;
    const grid = $(".form-grid", card);
    grid.append(
      field(
        {
          name: "name",
          type: "string",
          description: "A descriptive name for this processing step.",
        },
        data.name,
        (v) => {
          data.name = v || def.label;
          changed(false);
        },
        `${base}.name`,
      ),
    );
    grid.append(
      field(
        {
          name: "test",
          type: "boolean",
          description: "Run this step in test mode.",
        },
        data.test,
        (v) => {
          data.test = v;
          changed(false);
        },
        `${base}.test`,
      ),
    );
    for (const f of def?.fields || []) {
      const customField = category === "custom";
      const currentValue = customField ? data[f.name] : data.params?.[f.name];
      const fieldPath = customField ? `${base}.${f.name}` : `${base}.params.${f.name}`;
      grid.append(
        field(
          f,
          currentValue ?? f.default,
          (v) => {
            if (customField) {
              if (v == null || v === "") delete data[f.name];
              else data[f.name] = v;
            } else {
              data.params ??= {};
              if (v == null || v === "") delete data.params[f.name];
              else data.params[f.name] = v;
            }
            changed(false);
          },
          fieldPath,
        ),
      );
    }
    $(".remove", card).onclick = () => {
      state.config.steps.splice(index, 1);
      changed();
    };
    $(".up", card).onclick = () => move(index, -1);
    $(".down", card).onclick = () => move(index, 1);
    card.ondragstart = (e) => e.dataTransfer.setData("text/plain", String(index));
    card.ondragover = (e) => e.preventDefault();
    card.ondrop = (e) => {
      e.preventDefault();
      const from = Number(e.dataTransfer.getData("text/plain"));
      if (from === index) return;
      const [moving] = state.config.steps.splice(from, 1);
      state.config.steps.splice(index, 0, moving);
      changed();
    };
    root.append(card);
  });
}
function move(index, delta) {
  const target = index + delta;
  if (target < 0 || target >= state.config.steps.length) return;
  [state.config.steps[index], state.config.steps[target]] = [
    state.config.steps[target],
    state.config.steps[index],
  ];
  changed();
}
function setupAdd() {
  const category = $("#step-category"),
    mode = $("#step-mode");
  category.innerHTML = "";
  for (const c of state.registry.categories) {
    const o = document.createElement("option");
    o.value = c.name;
    o.textContent = c.label;
    category.append(o);
  }
  const modes = () => {
    mode.innerHTML = "";
    const current = state.registry.categories.find((c) => c.name === category.value);
    for (const m of current.modes) {
      const o = document.createElement("option");
      o.value = m.name;
      o.textContent = m.label;
      mode.append(o);
    }
  };
  category.onchange = modes;
  modes();
  $("#add-step").onclick = () => {
    const def = modeDef(category.value, mode.value);
    if (category.value === "custom") {
      const custom = { name: def.label, test: false };
      for (const f of def.fields) if (f.default != null) custom[f.name] = f.default;
      state.config.steps.push({ custom });
      changed();
      return;
    }
    const params = {};
    for (const f of def.fields) if (f.default != null) params[f.name] = f.default;
    state.config.steps.push({
      [category.value]: {
        name: def.label,
        mode: mode.value,
        test: false,
        params,
      },
    });
    changed();
  };
}
// -----------------------------------------------------------------------------
// Configuration synchronization and validation
// -----------------------------------------------------------------------------

async function changed(full = true) {
  if (full) {
    renderGeneral();
    renderGeneralSteps();
    renderSteps();
  }
  const [yamlResult, valid] = await Promise.all([
    api("/api/config/yaml", {
      method: "POST",
      body: JSON.stringify({ config: state.config }),
    }),
    api("/api/config/validate", {
      method: "POST",
      body: JSON.stringify({ config: state.config }),
    }),
  ]);
  $("#yaml").value = yamlResult.yaml;
  state.yamlLocations = yamlResult.locations;
  const badge = $("#validation");
  badge.textContent = valid.ok
    ? "Valid pipeline"
    : (valid.issues[0]?.message || "Invalid pipeline").split("\n")[0];
  badge.title = valid.issues.map((i) => i.message).join("\n");
  badge.className = "status " + (valid.ok ? "ok" : "bad");
  renderSummary();
}
function renderSummary() {
  const general = state.config.general;
  $("#summary").innerHTML =
    `<div class="summary-card"><b>${state.config.steps.length}</b><span>processing steps</span></div><div class="summary-card"><b>${general.sample_data ? label(general.sample_data) : "Custom"}</b><span>input source</span></div><div class="summary-card"><b>${general.n_jobs || 1}</b><span>worker${general.n_jobs === 1 ? "" : "s"}</span></div><div class="summary-card"><b>${general.track_changes === false ? "Off" : "On"}</b><span>step history</span></div>`;
}
// -----------------------------------------------------------------------------
// Workspace navigation, demos, and saved configurations
// -----------------------------------------------------------------------------

function tabs() {
  for (const button of $$(".tabs button"))
    button.onclick = () => {
      $$(".tabs button").forEach((b) => b.classList.toggle("active", b === button));
      $$("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== button.dataset.tab));
    };
}
async function loadDemos() {
  const demos = await api("/api/demos");
  const root = $("#demos");
  const groups = new Map();
  for (const demo of demos) {
    if (!groups.has(demo.group)) groups.set(demo.group, []);
    groups.get(demo.group).push(demo);
  }
  for (const [group, groupDemos] of groups) {
    const section = document.createElement("section");
    section.className = "demo-group";
    section.innerHTML = `<h3 class="demo-group-title">${escapeHtml(group)}</h3><div class="demo-group-grid"></div>`;
    const grid = $(".demo-group-grid", section);
    for (const demo of groupDemos) {
      const card = document.createElement("article");
      card.className = "demo-card";
      card.innerHTML = `<span class="dataset-badge">${escapeHtml(demo.dataset)}</span><h3>${escapeHtml(demo.title)}</h3><p>${escapeHtml(demo.description)}</p><div class="dataset-info"><b>Example dataset</b><span>${escapeHtml(demo.dataset_description)}</span></div>${demo.note ? `<div class="demo-note"><b>Before you run this</b><span>${escapeHtml(demo.note)}</span></div>` : ""}<button>Load demo</button>`;
      $("button", card).onclick = async () => {
        state.config = await api("/api/demos/" + demo.id);
        await changed();
        toast(`${demo.title} loaded`);
        $('.tabs button[data-tab="log"]').click();
      };
      grid.append(card);
    }
    root.append(section);
  }
}
async function refreshSaved() {
  const names = await api("/api/configs");
  const sel = $("#saved-configs");
  sel.innerHTML =
    '<option value="">Open saved…</option>' +
    names.map((n) => `<option value="${n}">${n}</option>`).join("");
}
function configActions() {
  $("#save").onclick = async () => {
    try {
      const name = $("#config-name").value.trim();
      if (!name) throw new Error("Enter a configuration name");
      await api("/api/configs", {
        method: "POST",
        body: JSON.stringify({ name, config: state.config }),
      });
      await refreshSaved();
      toast("Configuration saved");
    } catch (e) {
      toast(e.message);
    }
  };
  $("#saved-configs").onchange = async (e) => {
    if (!e.target.value) return;
    state.config = await api("/api/configs/" + encodeURIComponent(e.target.value));
    $("#config-name").value = e.target.value;
    await changed();
    toast("Configuration loaded");
  };
}
// -----------------------------------------------------------------------------
// Bidirectional YAML navigation
// -----------------------------------------------------------------------------

function lineOffset(text, line) {
  let offset = 0;
  for (let i = 0; i < line; i++) {
    const next = text.indexOf("\n", offset);
    if (next < 0) return text.length;
    offset = next + 1;
  }
  return offset;
}
function selectYamlPath(path) {
  const loc = state.yamlLocations.find((item) => item.path === path);
  if (!loc) return;
  const editor = $("#yaml"),
    start = lineOffset(editor.value, loc.start_line),
    endLine = lineOffset(editor.value, loc.end_line + 1);
  editor.setSelectionRange(start, Math.max(start, endLine - 1));
  editor.scrollTop = Math.max(0, loc.start_line * 20 - editor.clientHeight / 3);
}
function navigateFromYaml() {
  const editor = $("#yaml"),
    line = editor.value.slice(0, editor.selectionStart).split("\n").length - 1;
  const matches = state.yamlLocations
    .filter((item) => item.start_line <= line && item.end_line >= line)
    .sort((a, b) => b.path.length - a.path.length);
  if (!matches.length) return;
  let path = matches[0].path;
  let target = $$("[data-config-path]").find((el) => el.dataset.configPath === path);
  while (!target && path.includes(".")) {
    path = path.slice(0, path.lastIndexOf("."));
    target = $$("[data-config-path]").find((el) => el.dataset.configPath === path);
  }
  if (!target) return;
  const details = target.closest("details");
  if (details) details.open = true;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.remove("yaml-target");
  requestAnimationFrame(() => target.classList.add("yaml-target"));
  const input = target.querySelector("input,select,textarea");
  if (input) setTimeout(() => input.focus({ preventScroll: true }), 350);
}
let yamlTimer;
function yamlEditing() {
  const editor = $("#yaml");
  editor.addEventListener("click", navigateFromYaml);
  editor.addEventListener("input", () => {
    clearTimeout(yamlTimer);
    yamlTimer = setTimeout(async () => {
      try {
        const parsed = await api("/api/config/parse", {
          method: "POST",
          body: JSON.stringify({ yaml_content: editor.value }),
        });
        state.config = parsed.config;
        state.yamlLocations = parsed.locations;
        renderGeneral();
        renderGeneralSteps();
        renderSteps();
        renderSummary();
        const badge = $("#validation");
        badge.textContent = "Valid pipeline";
        badge.className = "status ok";
      } catch (e) {
        const badge = $("#validation");
        badge.textContent = "Invalid YAML";
        badge.title = e.message;
        badge.className = "status bad";
      }
    }, 450);
  });
}
// -----------------------------------------------------------------------------
// Logs and processed-image previews
// -----------------------------------------------------------------------------

function appendLog(message, cls = "") {
  const c = $("#console");
  const span = document.createElement("span");
  span.className = cls;
  span.textContent = (c.textContent.trim() ? "\n" : "") + message;
  c.append(span);
  c.scrollTop = c.scrollHeight;
}
function imageFigure(image, caption) {
  const fig = document.createElement("figure");
  fig.className = "image-card";
  fig.innerHTML = `<a href="${image.url}" target="_blank"><img loading="lazy" src="${image.url}" alt="${image.filename}"></a><figcaption title="${caption}">${caption} · ${image.width}×${image.height}</figcaption>`;
  return fig;
}
function renderPreviewStep() {
  const manifest = state.previewManifest,
    stepSelect = $("#image-step"),
    filename = $("#image-file").value,
    root = $("#images");
  root.innerHTML = "";
  if (filename) {
    let shown = 0;
    for (const step of manifest?.steps || []) {
      const image = step.images.find((item) => item.filename === filename);
      if (!image) continue;
      root.append(imageFigure(image, `Step ${step.index}: ${step.name}`));
      shown++;
    }
    $("#image-count").textContent =
      `${shown} pipeline stage${shown === 1 ? "" : "s"} · ${filename}`;
    if (!shown)
      root.innerHTML = '<p class="empty">This image is not available in the retained steps.</p>';
    return;
  }
  const step = manifest?.steps.find((item) => String(item.index) === stepSelect.value);
  if (!step) {
    root.innerHTML = '<p class="empty">No retained images are available for this step.</p>';
    return;
  }
  $("#image-count").textContent =
    `${step.count} image${step.count === 1 ? "" : "s"} · ${step.images.length} previewed`;
  for (const image of step.images) root.append(imageFigure(image, image.filename));
}
async function showPreviews() {
  if (!state.runId) return;
  try {
    const manifest = await api(`/api/runs/${state.runId}/previews`);
    state.previewManifest = manifest;
    const select = $("#image-step"),
      fileSelect = $("#image-file");
    select.innerHTML = "";
    for (const step of manifest.steps) {
      const option = document.createElement("option");
      option.value = step.index;
      option.textContent = `Step ${step.index}: ${step.name} (${step.count})`;
      select.append(option);
    }
    fileSelect.innerHTML = '<option value="">All images in selected step</option>';
    for (const filename of manifest.available_images ||
      manifest.steps.at(-1)?.images.map((item) => item.filename) ||
      []) {
      const option = document.createElement("option");
      option.value = filename;
      option.textContent = filename;
      fileSelect.append(option);
    }
    select.disabled = !manifest.steps.length;
    fileSelect.disabled = !manifest.steps.length;
    $("#save-images").disabled = !manifest.steps.length;
    select.onchange = renderPreviewStep;
    fileSelect.onchange = renderPreviewStep;
    const note = $("#history-note");
    note.hidden = manifest.track_changes;
    note.textContent = manifest.track_changes
      ? ""
      : "Step history was disabled for this run, so only the final output is available.";
    renderPreviewStep();
    await showMetadata();
    renderInvestigationPlots();
  } catch (e) {
    if (!String(e.message).includes("not ready")) toast(e.message);
  }
}

async function showMetadata() {
  if (!state.runId || !state.previewManifest?.metadata) return;
  const root = $("#metadata-table");
  const count = $("#metadata-count");
  const details = state.previewManifest.metadata;
  count.textContent = `${details.displayed} of ${details.count} rows · ${details.columns} columns`;
  try {
    const response = await fetch(`/api/runs/${state.runId}/metadata`);
    if (!response.ok) throw new Error("Metadata preview is not ready");
    root.innerHTML = await response.text();
  } catch (error) {
    root.innerHTML = '<p class="empty">The metadata preview could not be loaded.</p>';
    toast(error.message);
  }
}

function renderInvestigationPlots() {
  const plots = state.previewManifest?.investigations || [];
  const root = $("#investigation-plots");
  const count = $("#investigation-count");
  root.innerHTML = "";
  count.textContent = `${plots.length} plot${plots.length === 1 ? "" : "s"}`;
  if (!plots.length) {
    root.innerHTML = '<p class="empty">No investigation plots were generated during this run.</p>';
    return;
  }
  for (const plot of plots) {
    const figure = document.createElement("figure");
    figure.className = "investigation-card";
    figure.innerHTML = `<a href="${plot.url}" target="_blank" rel="noreferrer"><img loading="lazy" src="${plot.url}" alt="${escapeHtml(plot.title)}"></a><figcaption><span><b>${escapeHtml(plot.title)}</b><small>${escapeHtml(plot.step)}</small></span><a href="${plot.url}" download="${escapeHtml(plot.filename)}">Download</a></figcaption>`;
    root.append(figure);
  }
}
async function saveSelectedStep() {
  const button = $("#save-images"),
    step = Number($("#image-step").value);
  button.disabled = true;
  button.textContent = "Saving…";
  try {
    const started = await api("/api/export", {
      method: "POST",
      body: JSON.stringify({ run_id: state.runId, step }),
    });
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const result = await api("/api/exports/" + started.export_id);
      if (result.status === "complete") {
        toast(`Images saved to ${result.output_path}`);
        break;
      }
      if (result.status === "failed") {
        const error = result.messages.find((item) => item.event === "error");
        throw new Error(error?.message || "Image export failed");
      }
    }
  } catch (e) {
    toast(e.message);
  } finally {
    button.disabled = false;
    button.textContent = "Save images";
  }
}
// -----------------------------------------------------------------------------
// Memory chart and structured pipeline progress
// -----------------------------------------------------------------------------

const fmtMb = (value) =>
  value == null
    ? "–"
    : value >= 1024
      ? `${(value / 1024).toFixed(1)} GB`
      : `${Math.round(value)} MB`;
function resetRunDisplay() {
  state.memorySamples = [];
  state.memorySteps = [];
  $("#memory-meter").hidden = false;
  $("#memory-chart").innerHTML = "";
  $("#memory-current").textContent = "–";
  $("#memory-peak").textContent = "–";
  $("#memory-data").textContent = "–";
  $("#memory-time").textContent = "–";
  $("#general-run-status").textContent = "Pending";
  for (const card of $$(".step-card")) {
    card.classList.remove("run-running", "run-completed", "run-failed");
    $(".step-run-status", card).textContent = "Pending";
  }
}
function drawMemory() {
  const svg = $("#memory-chart"),
    samples = [...state.memorySamples, ...state.memorySteps].sort(
      (a, b) => a.elapsed_seconds - b.elapsed_seconds,
    );
  if (!samples.length) return;
  const width = Math.max(svg.clientWidth, 100),
    height = 42,
    pad = 3,
    peak = Math.max(...samples.map((item) => item.current_mb || 0), 1),
    last = Math.max(samples.at(-1).elapsed_seconds, 1),
    points = samples
      .map(
        (item) =>
          `${pad + (item.elapsed_seconds / last) * (width - 2 * pad)},${height - pad - (item.current_mb / peak) * (height - 2 * pad)}`,
      )
      .join(" ");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `<polygon class="mem-area" points="${pad},${height - pad} ${points} ${width - pad},${height - pad}"></polygon><polyline class="mem-line" points="${points}"></polyline>`;
  for (const step of state.memorySteps) {
    const x = pad + (step.elapsed_seconds / last) * (width - 2 * pad),
      y = height - pad - (step.current_mb / peak) * (height - 2 * pad),
      dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("class", "mem-dot");
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", y);
    dot.setAttribute("r", 3);
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${step.step_name}: peak ${fmtMb(step.step_peak_mb)}, settled ${fmtMb(step.current_mb)}, data ${fmtMb(step.dataset_mb)}`;
    dot.append(title);
    svg.append(dot);
  }
}
function handleMemory(event) {
  if (event.event === "memory_sample") state.memorySamples.push(event);
  else state.memorySteps.push(event);
  $("#memory-current").textContent = fmtMb(event.current_mb);
  $("#memory-peak").textContent = fmtMb(event.run_peak_mb);
  if (event.dataset_mb != null) $("#memory-data").textContent = fmtMb(event.dataset_mb);
  $("#memory-time").textContent = `${event.elapsed_seconds.toFixed(1)} s`;
  drawMemory();
}
function handleRunEvent(event) {
  const kind = event.type,
    index = event.step_index;
  if (!kind.startsWith("step_")) return;
  const card = index === 0 ? null : $(`.step-card[data-runtime-index="${index}"]`),
    status = index === 0 ? $("#general-run-status") : card?.querySelector(".step-run-status");
  if (!status) return;
  if (kind === "step_started") {
    status.textContent = "Running…";
    if (card) {
      card.classList.remove("run-completed", "run-failed");
      card.classList.add("run-running");
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  } else if (kind === "step_completed") {
    status.textContent = `✓ ${event.duration_seconds.toFixed(1)} s`;
    if (card) {
      card.classList.remove("run-running");
      card.classList.add("run-completed");
    }
  } else if (kind === "step_failed") {
    status.textContent = "Failed";
    if (card) {
      card.classList.remove("run-running");
      card.classList.add("run-failed");
      card.title = event.message || "";
    }
  }
}
// -----------------------------------------------------------------------------
// Pipeline execution and event streaming
// -----------------------------------------------------------------------------

function setBuilderLocked(locked) {
  for (const control of $$(".builder input,.builder select,.builder button"))
    control.disabled = locked;
  for (const card of $$(".step-card")) card.draggable = !locked;
  $("#yaml").readOnly = locked;
  $("#save").disabled = locked;
  $("#saved-configs").disabled = locked;
}
function streamEvents() {
  const source = new EventSource("/api/run/events");
  source.onmessage = async (e) => {
    const event = JSON.parse(e.data);
    if (event.event === "log") appendLog(event.message, event.level === "error" ? "log-error" : "");
    if (event.event === "status") {
      appendLog(event.message);
      $("#run-state").textContent = event.message;
    }
    if (event.event === "run_event") handleRunEvent(event);
    if (event.event === "memory_sample" || event.event === "memory_step") handleMemory(event);
    if (event.event === "previews_ready") await showPreviews();
    if (event.event === "complete") {
      appendLog(
        `Completed · previews ready for ${state.previewManifest?.steps.length || 0} retained step(s)`,
        "log-success",
      );
      $("#run-state").textContent = "Complete";
    }
    if (event.event === "error") {
      appendLog(`${event.error_type}: ${event.message}`, "log-error");
      $("#run-state").textContent = "Failed";
    }
    if (event.event === "closed") {
      $("#run").disabled = false;
      $("#stop").disabled = true;
      setBuilderLocked(false);
      await showPreviews();
      source.close();
    }
  };
  source.onerror = () => source.close();
}
function runActions() {
  $("#run").onclick = async () => {
    try {
      $('.tabs button[data-tab="log"]').click();
      $("#console").textContent = "";
      state.previewManifest = null;
      $("#investigation-count").textContent = "Running…";
      $("#investigation-plots").innerHTML =
        '<p class="empty">Investigation plots will appear here after this run.</p>';
      $("#metadata-count").textContent = "Loading…";
      $("#metadata-table").innerHTML =
        '<p class="empty">Image metadata will appear here after this run.</p>';
      resetRunDisplay();
      setBuilderLocked(true);
      $("#image-step").disabled = true;
      $("#image-file").disabled = true;
      $("#save-images").disabled = true;
      const result = await api("/api/run", {
        method: "POST",
        body: JSON.stringify({ config: state.config }),
      });
      state.runId = result.run_id;
      $("#run").disabled = true;
      $("#stop").disabled = false;
      $("#run-state").textContent = "Starting";
      appendLog(`Run ${state.runId} started`);
      streamEvents();
    } catch (e) {
      setBuilderLocked(false);
      toast(e.message);
    }
  };
  $("#stop").onclick = async () => {
    await api("/api/run/stop", { method: "POST" });
    $("#run-state").textContent = "Stopping";
  };
  $("#save-images").onclick = saveSelectedStep;
  $("#clear").onclick = () => {
    $("#console").textContent = "";
    state.previewManifest = null;
    $("#memory-meter").hidden = true;
    $("#image-step").innerHTML = "<option>No run yet</option>";
    $("#image-step").disabled = true;
    $("#image-file").innerHTML = '<option value="">All images</option>';
    $("#image-file").disabled = true;
    $("#save-images").disabled = true;
    $("#images").innerHTML =
      '<p class="empty">Processed previews will appear here after a successful run.</p>';
    $("#image-count").textContent = "No run yet";
    $("#metadata-count").textContent = "No run yet";
    $("#metadata-table").innerHTML =
      '<p class="empty">Image metadata will appear here after a successful run.</p>';
    $("#investigation-count").textContent = "No run yet";
    $("#investigation-plots").innerHTML =
      '<p class="empty">Investigation plots will appear here after a successful test-mode run.</p>';
    $("#history-note").hidden = true;
  };
}
// -----------------------------------------------------------------------------
// Application bootstrap
// -----------------------------------------------------------------------------

async function init() {
  tabs();
  configActions();
  runActions();
  yamlEditing();
  state.registry = await api("/api/registry");
  setupAdd();
  setupGeneralSteps();
  await Promise.all([changed(), loadDemos(), refreshSaved()]);
}
init().catch((e) => {
  console.error(e);
  toast(e.message);
});
