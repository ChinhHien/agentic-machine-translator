/* polyglot-lc web UI — one page, no framework, nothing fetched from a CDN.
 *
 * Everything here is built with `fetch` and `document.createElement`, and every
 * value from the server is written with `textContent`. Document names, step
 * names and provider errors are text a user typed or a log produced; none of it
 * is ever parsed as markup.
 *
 * Event shape (polyglot_lc.ui.events.Event): {ts, step, chunk_id, status,
 * latency, error}. Each step reports twice — `start`, then `ok`/`fail` — so the
 * step rail keeps the strongest state it has seen for a step name rather than
 * the last one.
 *
 * The page has no view of the server's configuration: a run is the document,
 * its two languages and the output format, and nothing else. The domain, the
 * glossary of terms and the chunk budget are the pipeline's own to work out,
 * so there is no settings screen and no key status here either.
 *
 * State attributes written here, and styled in app.css:
 *   .job[data-state]  .job-row[data-state]  .badge[data-state]
 *   .bar[data-state]  .step[data-state]     .pane[data-state]
 *   .alert[data-kind] .drop[data-state]
 */
"use strict";

const POLL_MS = 700;
const LIST_MS = 5000;
const FINAL_STATUSES = ["done", "failed", "stopped"];

const STATUS_WORDS = {
  empty: "chưa mở",
  running: "đang dịch",
  done: "xong",
  failed: "lỗi",
  stopped: "đã dừng",
};

/* A stage's state only ever moves up this ladder: a later `start` must not
   erase the `fail` a step already reported. */
const STEP_RANK = { wait: 0, running: 1, ok: 2, fail: 3 };

/* Stage names are the pipeline's own (`ui.step("segment")`); the words are only
   a courtesy. An unknown name is shown as it is. */
const STEP_WORDS = {
  ingest: "đọc tài liệu",
  detect_lang: "nhận ngôn ngữ",
  segment: "chia chunk",
  glossary: "áp từ điển",
  summary_tree: "tóm tắt",
  plan: "lập kế hoạch",
  analyse_chunks: "phân tích chunk",
  translate: "dịch",
  critic: "phản biện",
  refine: "tinh chỉnh",
  verify: "kiểm tra",
  retranslate: "dịch lại",
  save_trace: "lưu kinh nghiệm",
  consistency: "nhất quán",
  assemble_output: "ghép kết quả",
  skillopt_update: "cập nhật skill",
  run: "lượt chạy",
};

/* The pipeline's own closed set of error causes, in one word each. A cause the
   pipeline adds later shows as its raw name rather than as a blank badge. */
const CAUSE_WORDS = {
  rate_limit: "hết hạn mức",
  concurrency: "vượt song song",
  timeout: "quá thời gian",
  auth: "sai khoá",
  provider: "lỗi nhà cung cấp",
  parse: "phản hồi sai định dạng",
  verify: "không đạt kiểm tra",
  placeholder: "mất mốc mã",
  config: "sai cấu hình",
  input: "lỗi tài liệu",
  unknown: "chưa rõ",
};

const FILTERS = [
  ["all", "Tất cả"],
  ["running", "Đang dịch"],
  ["done", "Xong"],
  ["failed", "Lỗi"],
  ["stopped", "Đã dừng"],
];

const state = {
  runs: [],
  filter: "all",
  open: null, // run_id
  detail: null,
  events: [], // every event seen for the open run, in order
  seenStatus: null,
  next: 0,
  timer: null,
  listTimer: null,
  view: "split",
  hadEvents: false,
  panesFor: null, // run_id the pane states below belong to
  source: "idle", // idle | loading | done | failed
  target: "idle",
  sourceText: "",
  targetText: "",
  sourceNote: "",
  targetNote: "",
  sourceMeta: "",
  targetMeta: "",
  pendingOverwrite: "",
  uploaded: "", // the file just written to input/, which the dialog will run
};

/* ------------------------------------------------------------------ helpers */

function byId(id) {
  return document.getElementById(id);
}

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (content !== undefined) {
    node.textContent = content;
  }
  return node;
}

function clear(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
  return node;
}

function show(node, visible) {
  node.hidden = !visible;
  return node;
}

function setText(node, value) {
  node.textContent = value === undefined || value === null ? "" : String(value);
  return node;
}

function fill(node, values) {
  clear(node);
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") {
      node.append(value);
    }
  }
  return node;
}

function formatSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} kB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatCount(value) {
  return Number(value).toLocaleString("vi-VN");
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 60) {
    return `${minutes}:${String(rest).padStart(2, "0")}`;
  }
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function formatTime(epochSeconds) {
  const when = new Date((Number(epochSeconds) || 0) * 1000);
  const now = new Date();
  const minutes = (now.getTime() - when.getTime()) / 60000;
  const clock = `${String(when.getHours()).padStart(2, "0")}:${String(when.getMinutes()).padStart(2, "0")}`;
  if (minutes < 1) {
    return "vừa xong";
  }
  if (minutes < 60) {
    return `${Math.floor(minutes)} phút trước`;
  }
  if (when.toDateString() === now.toDateString()) {
    return clock;
  }
  return `${when.getDate()}/${when.getMonth() + 1} ${clock}`;
}

function isFinal(status) {
  return FINAL_STATUSES.indexOf(status) >= 0;
}

/* ---------------------------------------------------------------------- api */

async function request(method, path, body) {
  const init = { method: method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    throw new Error("không kết nối được máy chủ");
  }
  const raw = await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch (error) {
      payload = null;
    }
  }
  if (!response.ok) {
    const message =
      payload && payload.error
        ? payload.error
        : `máy chủ trả về ${response.status}`;
    const failure = new Error(message);
    failure.status = response.status;
    throw failure;
  }
  return payload;
}

function api(path) {
  return request("GET", path);
}

function post(path, body) {
  return request("POST", path, body === undefined ? {} : body);
}

async function readText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    let message = `máy chủ trả về ${response.status}`;
    try {
      const payload = JSON.parse(await response.text());
      if (payload && payload.error) {
        message = payload.error;
      }
    } catch (error) {
      message = `máy chủ trả về ${response.status}`;
    }
    const failure = new Error(message);
    failure.status = response.status;
    throw failure;
  }
  return response.text();
}

/* An upload with a progress bar, which `fetch` cannot report. */
function sendFile(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("PUT", path, true);
    request.setRequestHeader("Content-Type", "application/pdf");
    request.upload.addEventListener("progress", (event) => {
      if (onProgress && event.lengthComputable) {
        onProgress(event.loaded / event.total);
      }
    });
    request.addEventListener("load", () => {
      let payload = null;
      try {
        payload = JSON.parse(request.responseText);
      } catch (error) {
        payload = null;
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(payload);
        return;
      }
      const failure = new Error(
        payload && payload.error
          ? payload.error
          : `máy chủ trả về ${request.status}`,
      );
      failure.status = request.status;
      reject(failure);
    });
    request.addEventListener("error", () =>
      reject(new Error("không tải lên được")),
    );
    request.send(file);
  });
}

/* --------------------------------------------------------------- the banner */

function renderBanner(kind, message) {
  const banner = byId("banner");
  if (!message) {
    hideBanner();
    return;
  }
  banner.dataset.kind = kind;
  setText(banner, message);
  show(banner, true);
}

function hideBanner() {
  const banner = byId("banner");
  show(banner, false);
  setText(banner, "");
}

/* ------------------------------------------------------------ the job list */

/* The trash glyph is static markup, not server text, so it is built node by
   node: the page never splices an HTML string (server-derived content must stay
   plain text — a hard rule the web UI tests enforce). */
function trashIcon() {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "14");
  svg.setAttribute("height", "14");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(NS, "path");
  path.setAttribute(
    "d",
    "M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M6 6l1 14a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-14M10 11v6M14 11v6",
  );
  svg.appendChild(path);
  return svg;
}

function renderJobs() {
  const list = clear(byId("jobs"));
  const rows = state.runs.filter(
    (run) => state.filter === "all" || run.status === state.filter,
  );
  for (const run of rows) {
    const item = element("li", "job-item");
    const button = element("button", "job-row");
    button.type = "button";
    button.dataset.state = run.status;
    if (run.run_id === state.open) {
      button.setAttribute("aria-current", "true");
    }
    button.append(element("span", "job-row-name", run.file || run.run_id));
    button.append(element("span", "job-row-time", formatTime(run.started)));
    const sub = element("span", "job-row-sub");
    sub.append(element("span", null, `${run.src || "?"} → ${run.tgt || "?"}`));
    if (run.domain) {
      sub.append(element("span", null, run.domain));
    }
    if (run.chunks) {
      sub.append(element("span", null, `${formatCount(run.chunks)} chunk`));
    }
    sub.append(element("span", null, STATUS_WORDS[run.status] || run.status));
    button.append(sub);
    button.addEventListener("click", () => openJob(run.run_id));
    item.append(button);
    /* Erasing a run's history is destructive, so it asks first and it is the
       row's own action: a trash button that overlays the row's corner, hidden
       until hover. A delete can't be undone, so the confirm stays. */
    const del = element("button", "job-delete");
    del.type = "button";
    del.setAttribute("aria-label", `Xóa lượt dịch ${run.file || run.run_id}`);
    del.appendChild(trashIcon());
    del.addEventListener("click", () => deleteRun(run.run_id));
    item.append(del);
    list.append(item);
  }

  setText(
    byId("jobs-count"),
    `${formatCount(rows.length)} / ${formatCount(state.runs.length)}`,
  );
  const empty = byId("jobs-empty");
  if (rows.length) {
    show(empty, false);
  } else {
    setText(
      empty,
      state.runs.length
        ? "Không có lượt nào ở trạng thái này."
        : "Chưa có lượt dịch nào. Bấm “Dịch tài liệu mới” để bắt đầu.",
    );
    show(empty, true);
  }
}

function renderFilters() {
  const box = clear(byId("jobs-filter"));
  for (const entry of FILTERS) {
    const value = entry[0];
    const count =
      value === "all"
        ? state.runs.length
        : state.runs.filter((run) => run.status === value).length;
    const button = element(
      "button",
      "chip",
      `${entry[1]} ${formatCount(count)}`,
    );
    button.type = "button";
    button.setAttribute(
      "aria-pressed",
      value === state.filter ? "true" : "false",
    );
    button.addEventListener("click", () => {
      state.filter = value;
      renderFilters();
      renderJobs();
    });
    box.append(button);
  }
}

async function loadRuns() {
  clearTimeout(state.listTimer);
  let payload;
  try {
    payload = await api("/api/runs");
  } catch (error) {
    renderBanner(
      "error",
      `Không đọc được danh sách lượt dịch: ${error.message}`,
    );
    return;
  }
  state.runs = payload.runs || [];
  renderFilters();
  renderJobs();
  /* A finished run's status is only known from the list; when the open job's
     own status has moved on, the detail view has to catch up too. */
  if (state.open) {
    const summary = state.runs.filter((run) => run.run_id === state.open)[0];
    if (summary && state.detail && summary.status !== state.detail.run.status) {
      await loadDetail(state.open);
    }
  }
  if (state.runs.some((run) => run.status === "running")) {
    state.listTimer = setTimeout(loadRuns, LIST_MS);
  }
}

/* --------------------------------------------------------------- one job */

function actionLink(route, label, extra) {
  const link = element("a", "btn", label);
  link.href = route;
  if (extra) {
    link.title = extra;
  }
  return link;
}

function renderActions(detail) {
  const box = clear(byId("job-actions"));
  const run = detail.run;
  const base = `/api/runs/${encodeURIComponent(run.run_id)}`;
  for (const artifact of detail.artifacts || []) {
    const route = artifact.kind === "pdf" ? `${base}/pdf` : `${base}/output`;
    const label = artifact.kind === "pdf" ? "Tải PDF" : "Tải Markdown";
    const link = actionLink(
      `${route}?download=1`,
      `${label} (${formatSize(artifact.size)})`,
      artifact.name,
    );
    link.setAttribute("download", "");
    box.append(link);
  }
  if ((detail.artifacts || []).length > 1) {
    const bundle = actionLink(`${base}/bundle`, "Tải tất cả (.zip)");
    bundle.setAttribute("download", "");
    box.append(bundle);
  }
  if (detail.pdf) {
    const view = actionLink(`${base}/pdf`, "Mở bản PDF");
    view.target = "_blank";
    view.rel = "noreferrer";
    box.append(view);
  }
  if (run.status === "running") {
    const stop = element("button", "btn", "Dừng");
    stop.type = "button";
    stop.addEventListener("click", () => stopRun(run.run_id));
    box.append(stop);
  }
  if (run.status === "failed" || run.status === "stopped") {
    const resume = element("button", "btn", "Tiếp tục");
    resume.type = "button";
    resume.addEventListener("click", () => resumeRun(run.run_id));
    box.append(resume);
  }
  if (!box.childElementCount) {
    box.append(
      element(
        "p",
        "note",
        run.status === "running"
          ? "Bản dịch sẽ xuất hiện ở đây khi lượt dịch xong."
          : "Lượt dịch này không viết ra file nào.",
      ),
    );
  }
}

function renderJob(detail) {
  const run = detail.run;
  const job = byId("job");
  job.dataset.state = run.status;
  setText(byId("job-title"), run.file || run.run_id);
  const meta = clear(byId("job-meta"));
  const tags = [`${run.src || "?"} → ${run.tgt || "?"}`];
  if (run.domain) {
    tags.push(run.domain);
  }
  if (run.chunks) {
    tags.push(`${formatCount(run.chunks)} chunk`);
  }
  tags.push(formatTime(run.started));
  if (isFinal(run.status)) {
    tags.push(
      run.exit_code === 0 || run.exit_code === null
        ? STATUS_WORDS[run.status]
        : `mã thoát ${run.exit_code}`,
    );
  }
  for (const tag of tags) {
    meta.append(element("span", "tag", tag));
  }
  const badge = show(byId("job-status"), true);
  badge.dataset.state = run.status;
  setText(badge, STATUS_WORDS[run.status] || run.status);
  renderActions(detail);
  if (run.note) {
    const alert = byId("job-alert");
    alert.dataset.kind = run.status === "failed" ? "error" : "warn";
    setText(alert, run.note);
    show(alert, true);
  } else {
    show(byId("job-alert"), false);
  }
}

function renderEmptyJob() {
  const job = byId("job");
  job.dataset.state = "empty";
  setText(byId("job-title"), "Chưa mở lượt dịch nào");
  setText(
    byId("job-meta"),
    "Chọn một lượt ở cột bên trái, hoặc bắt đầu một lượt mới.",
  );
  /* No job open means no state to name: an empty badge is a blob, not a word. */
  const badge = show(byId("job-status"), false);
  badge.dataset.state = "empty";
  setText(badge, "");
  clear(byId("job-actions"));
  show(byId("job-alert"), false);
  show(byId("progress"), false);
  show(byId("problems"), false);
  paintPane(
    "pane-source",
    "source-text",
    "source-note",
    "source-meta",
    "empty",
    "Bản gốc của lượt dịch sẽ hiện ở đây.",
    "",
    "",
  );
  paintPane(
    "pane-target",
    "target-text",
    "target-note",
    "target-meta",
    "empty",
    "Bản dịch sẽ hiện ở đây.",
    "",
    "",
  );
}

/* --------------------------------------------------------------- progress */

function stepStateFor(status) {
  if (status === "start") {
    return "running";
  }
  if (status === "ok" || status === "patched") {
    return "ok";
  }
  if (status === "fail") {
    return "fail";
  }
  if (status === "retry") {
    return "running";
  }
  return ""; // `info` says something; it does not advance a step
}

function stepStates(events) {
  const order = [];
  const seen = new Map();
  for (const event of events) {
    const name = String(event.step || "");
    if (!name) {
      continue;
    }
    if (!seen.has(name)) {
      seen.set(name, "wait");
      order.push(name);
    }
    const mapped = stepStateFor(event.status);
    if (mapped && STEP_RANK[mapped] > STEP_RANK[seen.get(name)]) {
      seen.set(name, mapped);
    }
  }
  return order.map((name) => [name, seen.get(name)]);
}

function renderSteps(events) {
  const list = clear(byId("steps"));
  for (const entry of stepStates(events)) {
    const name = entry[0];
    const item = element("li", "step", STEP_WORDS[name] || name);
    item.dataset.state = entry[1];
    item.title = name;
    list.append(item);
  }
}

/* A chunk is finished once anything about it reached a terminal state, whether
   that was `ok` or `fail`: a failed chunk is not going to be retried by a later
   poll, so counting it as unfinished would freeze the bar forever. */
function chunksDone(events) {
  const done = new Set();
  for (const event of events) {
    if (!event.chunk_id) {
      continue;
    }
    if (
      event.status === "ok" ||
      event.status === "fail" ||
      event.status === "patched"
    ) {
      done.add(String(event.chunk_id));
    }
  }
  return done.size;
}

function renderProgress(detail) {
  const run = detail.run;
  const box = byId("progress");
  show(box, true);

  const bar = byId("bar");
  const fill = byId("bar-fill");
  const total = Number(run.chunks) || 0;
  const done = chunksDone(state.events);
  const percent = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
  /* `stopped` is not a failure, so its bar keeps the unfinished look and the
     text says what happened; `running` is the only state still moving. */
  const barState =
    run.status === "running"
      ? "running"
      : run.status === "done"
        ? "ok"
        : run.status === "failed"
          ? "fail"
          : "wait";
  bar.dataset.state = barState;
  fill.style.width = `${run.status === "done" ? 100 : percent}%`;
  bar.setAttribute(
    "aria-valuenow",
    String(run.status === "done" ? 100 : percent),
  );

  setText(
    byId("progress-count"),
    total
      ? `${formatCount(done)} / ${formatCount(total)} chunk`
      : `${formatCount(state.events.length)} sự kiện`,
  );

  const last = state.events.length
    ? state.events[state.events.length - 1]
    : null;
  const until =
    run.status === "running"
      ? Date.now() / 1000
      : last && last.ts
        ? Number(last.ts)
        : null;
  setText(
    byId("progress-elapsed"),
    until && run.started ? `đã chạy ${formatClock(until - run.started)}` : "",
  );

  setText(byId("progress-step"), progressWord(run, last));
  renderRoutes(detail);
  renderSteps(state.events);
}

/* The route mix, which is the answer to "why did this run cost so much": the
   server tallies it from the decisions the pipeline recorded. */
function renderRoutes(detail) {
  const routes = detail.routes || {};
  const total =
    (routes.fast || 0) + (routes.standard || 0) + (routes.deep || 0);
  setText(
    byId("progress-routes"),
    total
      ? `Đường đi: nhanh ${formatCount(routes.fast || 0)}` +
          ` · chuẩn ${formatCount(routes.standard || 0)}` +
          ` · sâu ${formatCount(routes.deep || 0)}`
      : "",
  );
}

function progressWord(run, last) {
  if (run.status === "running") {
    const step = last ? String(last.step || "") : "";
    return step ? `Đang chạy: ${STEP_WORDS[step] || step}` : "Đang chuẩn bị…";
  }
  if (run.status === "done") {
    return "Đã dịch xong.";
  }
  if (run.status === "failed") {
    return run.exit_code === null || run.exit_code === undefined
      ? "Lượt dịch đã lỗi."
      : `Lượt dịch đã lỗi (mã thoát ${run.exit_code}).`;
  }
  if (run.status === "stopped") {
    return "Đã dừng theo yêu cầu.";
  }
  return "";
}

/* The server only reports problems once per detail load; the page keeps its
   own copy from the events already polled, so a failing run shows its Vấn đề
   as they happen — the same feed that fills the progress bar. */
function problemsFromEvents(events) {
  const seen = new Map();
  for (const event of events) {
    const status = String(event.status || "");
    if (status !== "fail" && status !== "retry") {
      continue;
    }
    const step = String(event.step || "");
    const chunk = String(event.chunk_id || "");
    const key = `${step}::${chunk}`;
    const message = String(event.error || event.message || "");
    const existing = seen.get(key);
    if (!existing) {
      seen.set(key, {
        step: step,
        chunk_id: chunk,
        cause: String(event.cause || ""),
        message: message,
      });
    } else if (!existing.message && message) {
      existing.message = message;
    }
  }
  return Array.from(seen.values());
}

function renderProblems() {
  const card = byId("problems");
  const list = clear(byId("problems-list"));
  const byKey = new Map();
  for (const problem of problemsFromEvents(state.events)) {
    byKey.set(`${problem.step}::${problem.chunk_id}`, problem);
  }
  /* Server-side problems the event feed does not carry — chiefly the child's
     own message after a run dies before writing any event. */
  for (const problem of (state.detail && state.detail.problems) || []) {
    const key = `${problem.step}::${problem.chunk_id}`;
    if (!byKey.has(key)) {
      byKey.set(key, problem);
    }
  }
  for (const problem of byKey.values()) {
    const row = element("li", "problem");
    const where = problem.chunk_id
      ? `${problem.step} · ${problem.chunk_id}`
      : problem.step;
    row.append(element("span", "problem-where", where));
    if (problem.cause) {
      row.append(
        element(
          "span",
          "problem-cause",
          CAUSE_WORDS[problem.cause] || problem.cause,
        ),
      );
    }
    row.append(
      element("span", "problem-text", problem.message || "(không có mô tả)"),
    );
    list.append(row);
  }
  show(card, byKey.size > 0);
}

/* ------------------------------------------------------------------ panes */

function paintPane(
  paneId,
  textId,
  noteId,
  metaId,
  paneState,
  note,
  body,
  meta,
) {
  byId(paneId).dataset.state = paneState;
  setText(byId(noteId), note);
  setText(byId(textId), body);
  setText(byId(metaId), meta);
}

function renderPanes() {
  const detail = state.detail;
  if (!detail) {
    return;
  }
  const run = detail.run;
  const base = `/api/runs/${encodeURIComponent(run.run_id)}`;
  const source = { state: state.source, note: "", body: "", meta: "" };
  if (state.source === "idle" || state.source === "loading") {
    source.state = "loading";
    source.note = "Đang đọc bản gốc…";
  } else if (state.source === "failed") {
    source.state = "failed";
    source.note = state.sourceNote || "Không đọc được bản gốc.";
  } else {
    source.state = "done";
    source.body = state.sourceText || "";
    source.meta = state.sourceMeta || "";
  }
  paintPane(
    "pane-source",
    "source-text",
    "source-note",
    "source-meta",
    source.state,
    source.note,
    source.body,
    source.meta,
  );

  const target = { state: "empty", note: "", body: "", meta: "" };
  if (!detail.md) {
    if (run.status === "running") {
      target.note = "Bản dịch sẽ hiện ở đây khi lượt dịch xong.";
    } else if (run.status === "failed") {
      target.state = "failed";
      target.note = "Lượt dịch đã lỗi trước khi viết ra bản dịch.";
    } else if (run.status === "stopped") {
      /* Stopping is not failing: the pane is simply still empty, and says how
         it got that way. A red border here would claim a fault that did not
         happen. */
      target.note =
        "Lượt dịch đã dừng trước khi viết ra bản dịch — bấm “Tiếp tục” để dịch nốt.";
    } else {
      target.note = "Lượt dịch chưa có bản dịch.";
    }
  } else if (state.target === "idle" || state.target === "loading") {
    target.state = "loading";
    target.note = "Đang đọc bản dịch…";
  } else if (state.target === "failed") {
    target.state = "failed";
    target.note = state.targetNote || "Không đọc được bản dịch.";
  } else {
    target.state = "done";
    const written = (detail.artifacts || []).filter(
      (item) => item.kind === "md",
    )[0];
    /* The artefact's own name, never the path it happens to live at: where the
       repository keeps its files is not something a reader needs. */
    const name = written ? written.name : detail.md_path.split(/[\\/]/).pop();
    target.note = `File ${name} — hiển thị nguyên văn.`;
    target.body = state.targetText || "";
    target.meta = state.targetMeta || "";
  }
  paintPane(
    "pane-target",
    "target-text",
    "target-note",
    "target-meta",
    target.state,
    target.note,
    target.body,
    target.meta,
  );

  /* The panes are only read while they are visible; the source costs a whole
     extraction on the server the first time, so it waits until it is shown. */
  if (state.view !== "target" && state.source === "idle") {
    loadSource(base);
  }
  if (state.target === "idle" && detail.md) {
    loadTarget(base);
  }
}

async function loadSource(base) {
  state.source = "loading";
  renderPanes();
  try {
    const payload = await api(`${base}/source`);
    if (state.panesFor !== currentRunId()) {
      return;
    }
    state.sourceText = payload.text || "";
    state.sourceMeta = `${formatCount(payload.chars || 0)} ký tự · ${formatCount(payload.blocks || 0)} blocks`;
    state.source = "done";
  } catch (error) {
    if (state.panesFor !== currentRunId()) {
      return;
    }
    state.source = "failed";
    state.sourceNote = `Không đọc được bản gốc: ${error.message}`;
  }
  renderPanes();
}

async function loadTarget(base) {
  state.target = "loading";
  renderPanes();
  try {
    const body = await readText(`${base}/output`);
    if (state.panesFor !== currentRunId()) {
      return;
    }
    state.targetText = body;
    const words = body.trim() ? body.trim().split(/\s+/).length : 0;
    state.targetMeta = `${formatCount(words)} từ`;
    state.target = "done";
  } catch (error) {
    if (state.panesFor !== currentRunId()) {
      return;
    }
    state.target = "failed";
    state.targetNote = `Không đọc được bản dịch: ${error.message}`;
  }
  renderPanes();
}

function currentRunId() {
  return state.open;
}

function renderView() {
  byId("panes").dataset.view = state.view;
  const buttons = [
    ["view-split", "split"],
    ["view-source", "source"],
    ["view-target", "target"],
  ];
  for (const entry of buttons) {
    byId(entry[0]).setAttribute(
      "aria-pressed",
      entry[1] === state.view ? "true" : "false",
    );
  }
  renderPanes();
}

/* ----------------------------------------------------------------- polling */

async function loadDetail(runId) {
  const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
  if (state.open !== runId) {
    return;
  }
  state.detail = detail;
  renderJob(detail);
  renderProgress(detail);
  renderProblems();
  renderPanes();
  renderJobs();
}

async function openJob(runId) {
  clearTimeout(state.timer);
  clearTimeout(state.listTimer);
  hideBanner();
  if (state.open !== runId) {
    state.open = runId;
    state.detail = null;
    state.events = [];
    state.next = 0;
    state.seenStatus = null;
    state.hadEvents = false;
    state.panesFor = runId;
    state.source = "idle";
    state.target = "idle";
    state.sourceText = "";
    state.targetText = "";
    state.sourceNote = "";
    state.targetNote = "";
  }
  renderJobs();
  try {
    await loadDetail(runId);
  } catch (error) {
    renderBanner("error", `Không mở được lượt dịch: ${error.message}`);
    return;
  }
  const run = state.detail.run;
  state.seenStatus = run.status;
  if (!isFinal(run.status)) {
    pollRun(runId);
  } else if (!state.events.length) {
    /* A run that died before its first event has nothing to replay; its own
       output is the only explanation, and the server already put it in
       `problems`. */
    pollRun(runId);
  }
}

async function pollRun(runId) {
  clearTimeout(state.timer);
  let payload;
  try {
    payload = await api(
      `/api/runs/${encodeURIComponent(runId)}/events?after=${state.next}`,
    );
  } catch (error) {
    if (state.open === runId) {
      renderBanner("error", `Không theo dõi được lượt dịch: ${error.message}`);
    }
    return;
  }
  if (state.open !== runId) {
    return; // the reader moved to another run
  }
  state.next = Number(payload.next) || state.next;
  if (payload.events && payload.events.length) {
    state.events = state.events.concat(payload.events);
    renderProblems(); // Vấn đề cập nhật ngay trong lúc dịch, cùng nguồn với thanh tiến độ
  }

  const changed = payload.status !== state.seenStatus;
  state.seenStatus = payload.status;
  /* The first poll of a healthy run usually beats the child's own log, so the
     record briefly carries a note saying the run reports no progress. When the
     log appears that note is taken back on the server, and the only thing left
     holding the stale copy is this page: one refresh drops it. */
  const appeared = Boolean(payload.has_events) && !state.hadEvents;
  state.hadEvents = Boolean(payload.has_events);
  if (state.detail) {
    state.detail.run.status = payload.status;
    state.detail.run.exit_code = payload.exit_code;
    renderProgress(state.detail);
    if (changed || appeared) {
      if (changed && isFinal(payload.status)) {
        state.target = "idle"; // a new result may have appeared
      }
      await loadDetail(runId);
    }
  }

  if (payload.malformed) {
    renderBanner(
      "warn",
      `${formatCount(payload.malformed)} dòng trong log lượt dịch chưa đọc được; sẽ thử lại ở lần cập nhật sau.`,
    );
  } else if (!payload.has_events && isFinal(payload.status)) {
    /* A run still starting has simply not opened its log yet; only one that has
       already ended without ever writing one has something to explain. */
    renderBanner(
      "warn",
      "Lượt dịch đã kết thúc mà không ghi file sự kiện, nên không có tiến độ để đối chiếu.",
    );
  } else if (changed) {
    hideBanner();
  }

  if (!isFinal(payload.status)) {
    state.timer = setTimeout(() => pollRun(runId), POLL_MS);
    return;
  }
  await loadRuns();
}

/* Stopping takes effect at once, with no confirmation: the chunks that already
   finished stay in the run's checkpoints, so the work is not lost and the
   `Tiếp tục` button next to this one is the way back. A modal in front of a
   reversible action only teaches people to click through modals. */
async function stopRun(runId) {
  try {
    await post(`/api/runs/${encodeURIComponent(runId)}/stop`);
  } catch (error) {
    renderBanner("error", `Không dừng được lượt dịch: ${error.message}`);
    return;
  }
  await loadRuns();
  await loadDetail(runId);
}

/* The delete's guard, drawn in the app's own <dialog> rather than the browser's
   native confirm. Resolves true only when the reader explicitly chooses "Xóa";
   "Hủy", Escape, or the backdrop all resolve false — the same as cancelling. */
function confirmDelete() {
  const dlg = byId("confirm-delete");
  if (typeof dlg.showModal !== "function") {
    return Promise.resolve(false); // no modal support: never delete on a guess
  }
  return new Promise((resolve) => {
    const onClosed = () => {
      dlg.removeEventListener("close", onClosed);
      resolve(dlg.returnValue === "delete");
    };
    dlg.addEventListener("close", onClosed);
    dlg.showModal();
  });
}

async function deleteRun(runId) {
  if (!(await confirmDelete())) {
    return;
  }
  try {
    await request("DELETE", `/api/runs/${encodeURIComponent(runId)}`);
  } catch (error) {
    renderBanner("error", `Không xóa được lượt dịch: ${error.message}`);
    return;
  }
  /* The deleted run can no longer be watched or read, so the view returns to
     the empty state — and its timers must not keep polling a 404. */
  clearTimeout(state.timer);
  clearTimeout(state.listTimer);
  if (state.open === runId) {
    state.open = null;
    state.detail = null;
    state.events = [];
    state.next = 0;
    state.seenStatus = null;
    state.hadEvents = false;
    state.panesFor = null;
    state.source = "idle";
    state.target = "idle";
    state.sourceText = "";
    state.targetText = "";
    state.sourceNote = "";
    state.targetNote = "";
    renderEmptyJob();
  }
  hideBanner();
  await loadRuns();
}

/* The format a resumed run should write is inferred from what it already has:
   a run that produced both files asked for both, one with only a PDF asked for
   the PDF. The record does not carry the choice, and re-writing Markdown a
   reader did not ask for would be worse than asking for it twice. */
function resumeFormat(detail) {
  const kinds = (detail.artifacts || []).map((artifact) => artifact.kind);
  if (kinds.indexOf("md") >= 0 && kinds.indexOf("pdf") >= 0) {
    return "both";
  }
  return kinds.indexOf("pdf") >= 0 ? "pdf" : "md";
}

async function resumeRun(runId) {
  const detail =
    state.detail && state.detail.run.run_id === runId ? state.detail : null;
  const run = detail
    ? detail.run
    : state.runs.filter((row) => row.run_id === runId)[0];
  if (!run) {
    renderBanner("error", "Không tìm thấy lượt dịch để tiếp tục.");
    return;
  }
  let created;
  try {
    created = await post("/api/runs", {
      file: run.file,
      src: run.src || "auto",
      tgt: run.tgt || "vi",
      domain: run.domain || "general",
      format: detail ? resumeFormat(detail) : "md",
      resume: true,
    });
  } catch (error) {
    renderBanner("error", `Không tiếp tục được lượt dịch: ${error.message}`);
    return;
  }
  /* Resuming starts a fresh run — the id changes — so the page follows it
     rather than staying on the run it was resumed from. */
  await loadRuns();
  if (created && created.run_id) {
    await openJob(created.run_id);
  }
}

/* ------------------------------------------------------------ a new job */

function openNewJobDialog() {
  const dialog = byId("new-job");
  show(byId("new-job-alert"), false);
  setText(byId("new-job-alert"), "");
  byId("drop").dataset.state = "idle";
  setText(byId("upload-note"), "");
  show(byId("upload-bar"), false);
  /* Every visit starts from an empty drop zone: nothing carries over from the
     last dialog, so "Bắt đầu dịch" only ever runs the file just dropped. */
  state.pendingOverwrite = "";
  state.uploaded = "";
  byId("new-job-start").disabled = true;
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
}

function closeNewJobDialog() {
  const dialog = byId("new-job");
  if (typeof dialog.close === "function") {
    dialog.close();
  } else {
    dialog.removeAttribute("open");
  }
}

function formAlert(message, kind) {
  const alert = byId("new-job-alert");
  if (!message) {
    show(alert, false);
    return;
  }
  alert.dataset.kind = kind || "error";
  setText(alert, message);
  show(alert, true);
}

function dropMessage(state_, text) {
  const drop = byId("drop");
  drop.dataset.state = state_;
  setText(byId("upload-note"), text);
}

async function uploadFile(file, overwrite) {
  const name = String((file && file.name) || "").trim();
  if (!name) {
    return;
  }
  if (!/\.pdf$/i.test(name)) {
    dropMessage("bad", "Chỉ nhận file .pdf — file khác không dịch được.");
    return;
  }
  dropMessage("busy", `Đang tải ${name}…`);
  show(byId("upload-bar"), true);
  byId("upload-bar").dataset.state = "running";
  byId("upload-fill").style.width = "0%";
  const route = `/api/inputs/${encodeURIComponent(name)}${overwrite ? "?overwrite=1" : ""}`;
  try {
    await sendFile(route, file, (ratio) => {
      byId("upload-fill").style.width = `${Math.round(ratio * 100)}%`;
    });
    /* Overwriting is confirmed, never assumed: the launcher must not silently
       replace a document the reader already has. */
    state.pendingOverwrite = "";
    state.uploaded = name;
    byId("upload-bar").dataset.state = "ok";
    byId("upload-fill").style.width = "100%";
    dropMessage("idle", `Đã tải ${name} lên — bấm “Bắt đầu dịch”.`);
    byId("new-job-start").disabled = false;
  } catch (error) {
    byId("upload-bar").dataset.state = "fail";
    if (error.status === 409) {
      state.pendingOverwrite = name;
      dropMessage(
        "bad",
        `${name} đã có trong input/ — bấm lại (hoặc kéo lại) để ghi đè.`,
      );
      return;
    }
    dropMessage("bad", `Không tải lên được: ${error.message}`);
  }
}

function handleDropped(file) {
  if (!file) {
    return;
  }
  uploadFile(file, state.pendingOverwrite === file.name);
}

async function startRun(event) {
  if (event) {
    event.preventDefault();
  }
  /* The document is the one just dropped on the page: nothing else can start a
     run, so there is no stale choice to run by mistake. */
  const file = state.uploaded;
  if (!file) {
    formAlert("Hãy kéo thả một file PDF vào ô trên.", "warn");
    return;
  }
  const format = byId("format-pdf").checked
    ? "pdf"
    : byId("format-both").checked
      ? "both"
      : "md";
  const body = {
    file: file,
    src: byId("src").value || "auto",
    tgt: byId("tgt").value || "vi",
    format: format,
  };
  const button = byId("new-job-start");
  button.disabled = true;
  button.textContent = "Đang chuẩn bị…";
  formAlert("Đang đọc tài liệu và lập kế hoạch…", "ok");
  try {
    const created = await post("/api/runs", body);
    closeNewJobDialog();
    await loadRuns();
    await openJob(created.run_id);
  } catch (error) {
    formAlert(`Không bắt đầu được lượt dịch: ${error.message}`, "error");
  } finally {
    button.disabled = !state.uploaded;
    button.textContent = "Bắt đầu dịch";
  }
}

/* ------------------------------------------------------------------- wiring */

function wire() {
  byId("nav-new").addEventListener("click", openNewJobDialog);

  byId("new-job-cancel").addEventListener("click", closeNewJobDialog);
  byId("new-job-form").addEventListener("submit", startRun);

  const drop = byId("drop");
  const input = byId("file-input");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      handleDropped(input.files[0]);
      input.value = "";
    }
  });
  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      if (drop.dataset.state !== "busy") {
        drop.dataset.state = "over";
      }
    });
  }
  drop.addEventListener("dragleave", () => {
    if (drop.dataset.state === "over") {
      drop.dataset.state = "idle";
    }
  });
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    const files = event.dataTransfer ? event.dataTransfer.files : null;
    handleDropped(files && files.length ? files[0] : null);
  });
  /* A file dropped anywhere else would otherwise be opened by the browser and
     the page would be replaced by a PDF. */
  for (const name of ["dragover", "drop"]) {
    window.addEventListener(name, (event) => {
      if (!byId("new-job").open) {
        event.preventDefault();
      }
    });
  }

  byId("view-split").addEventListener("click", () => {
    state.view = "split";
    renderView();
  });
  byId("view-source").addEventListener("click", () => {
    state.view = "source";
    renderView();
  });
  byId("view-target").addEventListener("click", () => {
    state.view = "target";
    renderView();
  });
}

function boot() {
  wire();
  renderEmptyJob();
  renderFilters();
  renderView();
  loadRuns();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
