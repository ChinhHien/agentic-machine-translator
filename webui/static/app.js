/* polyglot-lc web UI.
 *
 * Everything here is built with `fetch` and `document.createElement` plus
 * `textContent`. Nothing from the server is ever interpolated into markup:
 * step names, document names and error text all come from the user's own
 * files and logs, so they are text, never HTML.
 *
 * Event shape (see polyglot_lc.ui.events.Event): {ts, step, chunk_id, status,
 * latency, error}. A step with a chunk_id belongs to one chunk and fills a
 * cell in the matrix; a step without one advances the stage rail.
 */
"use strict";

const POLL_MS = 700;
const FINAL_STATUSES = ["done", "failed", "stopped"];

/* Chunk cells never go "backwards": a later ok must not erase an earlier fail. */
const CELL_RANK = { wait: 0, running: 1, ok: 2, patched: 3, retry: 4, fail: 5 };
const CELL_WORDS = {
  wait: "chờ",
  running: "đang chạy",
  ok: "xong",
  patched: "đã vá thuật ngữ",
  retry: "phải thử lại",
  fail: "lỗi",
};
const STATUS_WORDS = {
  running: "đang chạy",
  done: "xong",
  failed: "lỗi",
  stopped: "đã dừng",
};

/* Every step reports twice — once when it starts, once when it ends — so the
 * row has to say which of the two it is, or the log reads as duplicates. */
const EVENT_WORDS = {
  start: "bắt đầu",
  ok: "xong",
  fail: "lỗi",
  retry: "thử lại",
  info: "ghi chú",
};

const state = {
  runId: null,
  after: 0,
  total: 0,
  polling: false,
  positions: new Map(), // chunk id -> cell index
  used: [],
  cells: [],
  stages: new Map(), // stage step -> element
  seenStages: [],
};

const byId = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ utils */

async function api(path, options) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    const reason = (payload && payload.error) || response.statusText;
    throw new Error(reason);
  }
  return payload;
}

function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

function put(path, body) {
  return api(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clock(seconds) {
  const when = seconds ? new Date(seconds * 1000) : new Date();
  return when.toTimeString().slice(0, 8);
}

function duration(milliseconds) {
  const ms = Number(milliseconds) || 0;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function bytes(size) {
  const kb = Number(size) / 1024;
  return kb >= 1024 ? `${(kb / 1024).toFixed(1)}MB` : `${Math.max(1, Math.round(kb))}KB`;
}

/* ----------------------------------------------------------------- banner */

function renderBanner(payload) {
  const banner = byId("banner");
  if (!payload || !payload.text) {
    banner.hidden = true;
    banner.textContent = "";
    return;
  }
  banner.className = "banner";
  banner.dataset.kind = payload.kind || "warn";
  banner.textContent = payload.text;
  banner.hidden = false;
}

/* ------------------------------------------------------------ stage rail */

/** Stages are shown in the order the run reached them, not a guessed list:
 *  the pipeline's own step names are the only truth available here. */
function stageFor(step) {
  if (state.stages.has(step)) return state.stages.get(step);
  const item = element("li", "stage", step);
  item.dataset.state = "wait";
  state.stages.set(step, item);
  state.seenStages.push(step);
  byId("stage-rail").append(item);
  return item;
}

function renderStages(events) {
  for (const event of events) {
    if (event.chunk_id) continue;
    const item = stageFor(event.step);
    if (event.status === "fail") {
      item.dataset.state = "fail";
    } else if (event.status === "ok") {
      item.dataset.state = "ok";
    } else if (event.status === "start") {
      item.dataset.state = "running";
    }
  }
}

/* ---------------------------------------------------------- chunk matrix */

function positionFor(chunkId) {
  if (state.positions.has(chunkId)) return state.positions.get(chunkId);
  const match = /(\d+)\s*$/.exec(chunkId);
  let index = match ? Number(match[1]) - 1 : -1;
  if (index < 0 || index >= state.used.length || state.used[index]) {
    index = state.used.indexOf(false);
    if (index < 0) index = state.used.push(false) - 1;
  }
  state.used[index] = true;
  state.positions.set(chunkId, index);
  return index;
}

function cellAt(index) {
  const matrix = byId("chunk-matrix");
  while (state.cells.length <= index) {
    const cell = element("span", "cell");
    cell.dataset.state = "wait";
    cell.dataset.index = String(state.cells.length + 1);
    cell.setAttribute("role", "listitem");
    cell.title = `chunk ${state.cells.length + 1}: chờ`;
    state.cells.push(cell);
    state.used.push(false);
    matrix.append(cell);
  }
  return state.cells[index];
}

function setCell(index, next) {
  const cell = cellAt(index);
  const current = cell.dataset.state;
  if (CELL_RANK[next] < CELL_RANK[current]) return;
  cell.dataset.state = next;
  const name = `chunk ${index + 1}: ${CELL_WORDS[next]}`;
  cell.title = name;
  cell.setAttribute("aria-label", name);
}

function renderChunks(events, total) {
  state.total = Math.max(state.total, Number(total) || 0);
  for (const event of events) {
    if (!event.chunk_id) continue;
    const index = positionFor(event.chunk_id);
    if (event.status === "start") setCell(index, "running");
    else if (event.status === "ok") setCell(index, "ok");
    else if (event.status === "retry") setCell(index, "retry");
    else if (event.status === "fail") setCell(index, "fail");
    else if (event.status === "info") setCell(index, "patched");
  }
  const done = state.cells.filter((cell) => cell.dataset.state !== "wait").length;
  const planned = Math.max(state.total, state.cells.length);
  byId("chunk-summary").textContent = planned ? `${done}/${planned}` : "";
}

/* ---------------------------------------------------------------- events */

function eventRow(event) {
  const status = event.status || "info";
  const row = element("li", "event");
  row.dataset.status = status;
  row.append(
    element("span", "event-time", clock(event.ts)),
    element("span", "event-state", EVENT_WORDS[status] || status),
    element("span", "event-step", event.chunk_id ? `${event.step} ${event.chunk_id}` : event.step),
  );
  row.append(element("span", event.error ? "event-note" : "event-id", event.error || ""));
  row.append(element("span", "event-latency", event.latency ? duration(event.latency) : ""));
  return row;
}

function renderEvents(events) {
  if (!events.length) return;
  const list = byId("events");
  for (const event of events) list.append(eventRow(event));
  list.scrollTop = list.scrollHeight;
}

/* ------------------------------------------------------------------- runs */

async function loadHistory() {
  const data = await api("/api/runs");
  const list = byId("run-list");
  list.replaceChildren();
  for (const run of data.runs) {
    const item = element("li");
    const button = element("button", "run");
    button.type = "button";
    button.dataset.state = run.status;
    if (run.run_id === state.runId) button.setAttribute("aria-current", "true");
    button.append(
      element("span", "run-time", clock(run.started)),
      element("span", "run-name", run.file || run.run_id),
      element("span", "run-meta", run.chunks ? String(run.chunks) : ""),
    );
    button.title = `${STATUS_WORDS[run.status] || run.status} · ${run.run_id}`;
    button.addEventListener("click", () => selectRun(run.run_id));
    item.append(button);
    list.append(item);
  }
}

function newRunPanes(runId, total) {
  state.runId = runId;
  state.after = 0;
  state.total = Number(total) || 0;
  state.positions = new Map();
  state.used = [];
  state.cells = [];
  state.stages = new Map();
  state.seenStages = [];
  byId("stage-rail").replaceChildren();
  byId("chunk-matrix").replaceChildren();
  byId("events").replaceChildren();
  byId("chunk-summary").textContent = "";
  byId("stop-button").disabled = false;
  byId("run-hint").textContent = `Lượt chạy ${runId} đang chạy.`;
  renderBanner(null);
}

async function selectRun(runId) {
  state.polling = false; // a new selection takes over the single poll loop
  const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
  const run = detail.run;
  state.runId = runId;
  state.after = 0;
  state.total = run.chunks;
  state.positions = new Map();
  state.used = [];
  state.cells = [];
  state.stages = new Map();
  byId("stage-rail").replaceChildren();
  byId("chunk-matrix").replaceChildren();
  byId("events").replaceChildren();

  await showOutput(runId);
  const data = await api(`/api/runs/${encodeURIComponent(runId)}/events?after=0`);
  renderStages(data.events);
  renderChunks(data.events, run.chunks);
  renderEvents(data.events);
  state.after = data.next;
  byId("run-hint").textContent =
    `Lượt chạy ${runId}: ${STATUS_WORDS[data.status] || data.status}`;
  byId("stop-button").disabled = data.status !== "running";
  announceEvents(data);
  if (data.status === "running") {
    state.polling = true;
    setTimeout(() => pollRun(runId), POLL_MS);
  }
}

async function showOutput(runId) {
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/output`);
    const proof = byId("proof");
    if (!response.ok) {
      proof.dataset.state = "empty";
      proof.textContent = "Lượt chạy này chưa có file kết quả.";
      byId("proof-path").textContent = "";
      return;
    }
    proof.dataset.state = "done";
    proof.textContent = await response.text();
    const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
    byId("proof-path").textContent = detail.run.output_path || "";
  } catch (error) {
    renderBanner({ kind: "error", text: `Không đọc được kết quả: ${error.message}` });
  }
}

/** Tell the user when a run has no event log at all, instead of showing an
 *  empty list that looks like the run is doing nothing. */
function announceEvents(data) {
  if (data.has_events === false) {
    renderBanner({
      kind: "warn",
      text: "Lượt chạy này không ghi file sự kiện, nên không có tiến độ để hiện. "
        + "Bật LOG_JSONL=1 trong .env rồi chạy lại để theo dõi từng chunk.",
    });
  } else if (data.malformed) {
    renderBanner({
      kind: "warn",
      text: `File log có ${data.malformed} dòng hỏng; các dòng còn lại vẫn được hiển thị.`,
    });
  } else {
    renderBanner(null);
  }
}

async function pollRun(runId) {
  if (state.runId !== runId) return; // the user moved on
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/events?after=${state.after}`);
    state.after = data.next;
    renderStages(data.events);
    renderChunks(data.events, state.total);
    renderEvents(data.events);
    announceEvents(data);
    if (FINAL_STATUSES.includes(data.status)) {
      state.polling = false;
      byId("stop-button").disabled = true;
      byId("run-hint").textContent =
        `Lượt chạy ${runId}: ${STATUS_WORDS[data.status] || data.status}`;
      await showOutput(runId);
      await loadHistory();
      if (data.status === "failed") {
        renderBanner({ kind: "error", text: "Lượt chạy kết thúc với lỗi. Xem dòng sự kiện màu đỏ." });
      }
      return;
    }
    setTimeout(() => pollRun(runId), POLL_MS);
  } catch (error) {
    state.polling = false;
    renderBanner({ kind: "error", text: `Mất liên lạc với lượt chạy: ${error.message}` });
  }
}

async function startRun() {
  const file = byId("file").value;
  if (!file) {
    renderBanner({ kind: "error", text: "Chọn một tài liệu trong input/ trước đã." });
    return;
  }
  const tokens = byId("max-tokens").value.trim();
  byId("run-button").disabled = true;
  try {
    const started = await post("/api/runs", {
      file: file,
      src: byId("src").value.trim() || "auto",
      tgt: byId("tgt").value.trim() || "vi",
      domain: byId("domain").value.trim() || "general",
      glossary: byId("glossary").value || null,
      max_tokens: tokens ? Number(tokens) : null,
      mock: byId("mock").checked,
    });
    newRunPanes(started.run_id, started.chunks);
    state.polling = true;
    setTimeout(() => pollRun(started.run_id), POLL_MS);
  } catch (error) {
    renderBanner({ kind: "error", text: `Không chạy được: ${error.message}` });
  } finally {
    byId("run-button").disabled = false;
  }
}

async function stopRun() {
  if (!state.runId) return;
  try {
    const stopped = await post(`/api/runs/${encodeURIComponent(state.runId)}/stop`);
    renderBanner({ kind: "warn", text: `Lượt chạy ${STATUS_WORDS[stopped.status] || stopped.status}.` });
    state.polling = false;
    byId("stop-button").disabled = true;
  } catch (error) {
    renderBanner({ kind: "error", text: `Không dừng được: ${error.message}` });
  }
}

/* ---------------------------------------------------------------- inputs */

async function loadInputs() {
  const data = await api("/api/inputs");
  const select = byId("file");
  select.replaceChildren();
  for (const file of data.files) {
    const option = element("option", null, `${file.name} · ${bytes(file.size)}`);
    option.value = file.name;
    select.append(option);
  }
  if (!data.files.length) {
    const option = element("option", null, "chưa có tài liệu nào trong input/");
    option.value = "";
    select.append(option);
  }
}

async function loadGlossaries() {
  const data = await api("/api/glossaries");
  const select = byId("glossary");
  select.replaceChildren();
  const none = element("option", null, "không dùng");
  none.value = "";
  select.append(none);
  for (const file of data.files) {
    const option = element("option", null, `${file.name} · ${file.entries} cặp`);
    option.value = file.name;
    select.append(option);
  }
  const list = byId("glossary-list");
  list.replaceChildren();
  for (const file of data.files) {
    list.append(element("li", null, `${file.name} · ${file.entries} cặp`));
  }
}

function parsePairs(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length)
    .map((line) => line.split(",").map((cell) => cell.trim()));
}

async function saveGlossary() {
  const name = byId("glossary-name").value.trim();
  const pairs = parsePairs(byId("glossary-pairs").value);
  if (!name) {
    renderBanner({ kind: "error", text: "Đặt tên file glossary trước đã, ví dụ legal.csv." });
    return;
  }
  try {
    await put(`/api/glossaries/${encodeURIComponent(name)}`, { pairs: pairs });
    renderBanner({ kind: "warn", text: `Đã lưu ${name} với ${pairs.length} cặp.` });
    await loadGlossaries();
  } catch (error) {
    renderBanner({ kind: "error", text: `Không lưu được glossary: ${error.message}` });
  }
}

async function loadSkills() {
  const data = await api("/api/skills");
  const list = byId("skills-list");
  list.replaceChildren();
  if (!data.skills.length) {
    list.append(element("li", null, "chưa có skill nào được học"));
    return;
  }
  for (const skill of data.skills) {
    const score = Number(skill.mean_score || 0).toFixed(1);
    list.append(element("li", null, `${skill.pair} · ${skill.domain} · điểm ${score}`));
  }
}

/* ---------------------------------------------------------------- config */

const configInputs = new Map();

function renderConfig(data) {
  const fields = byId("config-fields");
  fields.replaceChildren();
  configInputs.clear();

  for (const key of data.editable) {
    const value = data.values ? data.values[key] : undefined;
    const field = element("div", "field");
    const label = element("label", null, key);
    label.setAttribute("for", `cfg-${key}`);
    let input;
    if (typeof value === "boolean") {
      input = element("input");
      input.type = "checkbox";
      input.checked = value;
      label.className = "check";
      label.append(input, element("span", null, key));
    } else {
      input = element("input");
      input.value = value === undefined || value === null ? "" : String(value);
      field.append(label);
    }
    input.id = `cfg-${key}`;
    configInputs.set(key, input);
    field.append(input);
    const source = data.sources ? data.sources[key] : "";
    if (source) {
      const note = element("span", "hint", source === "env"
        ? "biến môi trường đang thắng; sửa .env sẽ không có hiệu lực cho tới khi bỏ biến đó"
        : `nguồn: ${source}`);
      field.append(note);
    }
    fields.append(field);
  }

  const readonly = element("p", "hint", `Chỉ đọc: ${(data.readonly || []).join(", ")}`);
  fields.append(readonly);

  const secrets = Object.entries(data.secrets || {});
  if (secrets.length) {
    const text = secrets.map(([name, status]) => `${name}: ${status === "set" ? "đã đặt" : "chưa đặt"}`);
    fields.append(element("p", "hint", `API key (không hiện giá trị): ${text.join(" · ")}`));
  }
  byId("config-note").textContent = data.probe_error
    ? `Không đọc được cấu hình hiệu lực: ${data.probe_error}`
    : "Giá trị dưới đây là cấu hình hiệu lực, không phải giá trị mặc định.";
}

async function loadConfig() {
  try {
    renderConfig(await api("/api/config"));
  } catch (error) {
    byId("config-note").textContent = `Không đọc được cấu hình: ${error.message}`;
  }
}

async function saveConfig() {
  const values = {};
  for (const [key, input] of configInputs) {
    const field = input.type === "checkbox" ? input.checked : input.value;
    values[key] = input.type === "checkbox" ? field : interpret(field);
  }
  try {
    const result = await put("/api/config", { values: values });
    const saved = result.saved.length ? `đã lưu: ${result.saved.join(", ")}` : "không lưu gì";
    const rejected = result.rejected.map((item) => `${item.key} (${item.reason})`).join(", ");
    renderBanner({
      kind: rejected ? "warn" : "warn",
      text: rejected ? `${saved}. Bị từ chối: ${rejected}` : saved,
    });
    await loadConfig();
  } catch (error) {
    renderBanner({ kind: "error", text: `Không lưu được cấu hình: ${error.message}` });
  }
}

/** Settings are stored as text; keep a number looking like a number and a
 *  bare true/false looking like the 1/0 the config loader expects. */
function interpret(text) {
  const trimmed = String(text).trim();
  if (trimmed === "true") return 1;
  if (trimmed === "false") return 0;
  if (trimmed !== "" && !Number.isNaN(Number(trimmed))) return Number(trimmed);
  return trimmed;
}

/* ------------------------------------------------------------- bootstrap */

async function boot() {
  byId("run-button").addEventListener("click", startRun);
  byId("stop-button").addEventListener("click", stopRun);
  byId("config-save").addEventListener("click", saveConfig);
  byId("glossary-save").addEventListener("click", saveGlossary);
  try {
    await Promise.all([loadInputs(), loadHistory(), loadGlossaries(), loadSkills()]);
  } catch (error) {
    renderBanner({ kind: "error", text: `Không tải được dữ liệu: ${error.message}` });
  }
  await loadConfig();
}

document.addEventListener("DOMContentLoaded", boot);
