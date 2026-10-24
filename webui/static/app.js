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
 * its two languages and the output format, and nothing else. The glossary of
 * terms and the chunk budget are the pipeline's own to work out,
 * so there is no settings screen and no key status here either.
 *
 * State attributes written here, and styled in app.css:
 *   .job[data-state]  .job-row[data-state]  .badge[data-state]
 *   .bar[data-state]  .step[data-state]     .pane[data-state]
 *   .alert[data-kind] .drop[data-state]     .chunk[data-state]
 *   .chunk[data-recovered]   a chunk the run walked past after a failure
 *   .log-line[data-expanded] a log line opened to its full message and fields
 */
"use strict";

const POLL_MS = 700;
const LIST_MS = 5000;
const FINAL_STATUSES = ["done", "failed", "stopped"];

/*: The folded library has room for one mark per document and no words, so the
   state is a shape first and a colour second: a running run is a dot, a
   finished one a tick, a failed one a bang, a stopped one a hold. Colour alone
   would leave a reader who cannot separate the four tones with nothing. */
const STATE_MARKS = {
  running: "●",
  done: "✓",
  failed: "!",
  stopped: "‖",
};

const STATUS_WORDS = {
  empty: "chưa mở",
  running: "đang dịch",
  done: "xong",
  failed: "lỗi",
  stopped: "đã dừng",
};

/* What one phase's label says, from the state its own events leave it in.

   A phase is visited once per chunk (`ui.step("translate", chunk.id)`), so one
   run reports on the same phase hundreds of times and there is no honest way to
   follow it chunk by chunk. The rail keeps only the two verdicts that are
   final: `ok` once the whole input has gone through the phase, `fail` once the
   run has ended on it. Until then a phase shown here is the one being worked
   on, and nothing more is claimed about it. */
const STEP_STATE_WORDS = {
  running: "đang chạy",
  ok: "xong",
  fail: "lỗi",
};

/* Stage names are the pipeline's own (`ui.step("segment")`): `step` in a log
   line is one of these keys, and this map is the only place it becomes
   Vietnamese. The words are a courtesy — an unknown name is shown as it is. */
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
  collect: "chốt bản dịch",
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

/* What each field of a log event is called in the reader's language, for the
   detail panel a log line opens to. A key this map has not met shows under its
   own name rather than not at all. */
const LOG_DETAIL_WORDS = {
  type: "bản ghi",
  run_id: "mã lượt chạy",
  latency: "độ trễ",
  duration_ms: "thời gian chạy",
  tokens: "token",
  active_api_calls: "luồng API",
  max_active_api_calls: "luồng API cao nhất",
  cause: "nguyên nhân",
  retries: "lượt thử lại",
  retranslated: "lần dịch lại",
  exit_code: "mã thoát",
  chunks: "số chunk",
  chunks_done: "chunk đã dịch",
  chunks_failed: "chunk lỗi",
  resume_requested: "yêu cầu tiếp tục",
  resumed_chunks: "chunk dùng lại",
  checkpoint_usable: "checkpoint dùng được",
  doc_key: "khóa tài liệu",
  plan_hash: "băm kế hoạch",
  max_tokens: "ngân sách token",
  version: "phiên bản",
  sha256: "sha256",
  bytes: "kích thước",
  pages: "số trang",
  src: "ngôn ngữ nguồn",
  tgt: "ngôn ngữ đích",
  flags: "cờ",
  agents: "agent",
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
  view: "split", // split | source | target
  /* The workspace's own tools: whether the two columns travel together, whether
     the pair takes the whole window, and how the pair is shared (the left
     column's percentage, moved by dragging the divider between them). */
  sync: true,
  zoom: false,
  split: 50,
  /* Where each column was last left. Hiding a column takes its scroll position
     with it, so the reader's place is held here and handed back to it. */
  scroll: { source: 0, target: 0 },
  /* The library: a column with room for one (expanded | collapsed), or a drawer
     over the page on a screen without, where what matters is whether it is
     open. The drawer is read from the window, then kept up to date with it. */
  rail: "expanded",
  railOpen: false,
  drawer: false,
  tab: "result", // which detail tab is open: the panes, the chunk list or the log
  /* How much of `events` the log has already drawn, so a poll appends only what
     is new. A run of a few thousand events must not rebuild the whole list on
     every tick, and `loggedProblems` keeps the server's own notes — the ones no
     event carries — from being drawn twice. */
  logged: 0,
  loggedProblems: new Set(),
  /* Whether the chunk list is still following the run: scrolling anywhere else
     is the reader claiming their place, and it is theirs until they come back
     to the run's edge. */
  follow: true,
  /* Which rows the chunk list shows ("all" or "fail"), and which one chunk's
     message is open. Both belong to the open run, so opening another starts
     from the whole list with nothing expanded. */
  chunkFilter: "all",
  chunkOpen: null,
  hadEvents: false,
  /* The resume notice that has already been raised: the resume record it
     belongs to, by its time. A notice is shown once for the attempt it
     describes, not on every poll of a run that goes on for an hour. */
  resumeSeen: 0,
  panesFor: null, // run_id the pane states below belong to
  source: "idle", // idle | loading | done | failed
  target: "idle",
  /* The translation so far, read from the run's own chunks on disk while it is
     still going: their text in the document's order, how many there are, and
     how many the run had finished when they were last asked for — so a poll
     that changed nothing costs no request at all. */
  partialText: "",
  partialPieces: 0,
  partialDone: -1,
  sourceText: "",
  targetText: "",
  sourceNote: "",
  targetNote: "",
  sourceMeta: "",
  targetMeta: "",
  uploaded: "", // the file just written to input/, which the dialog will run
  bannerTimer: null, // pending auto-hide of the notice bar
  bannerFade: null, // pending removal once the fade-out has run
  splashFade: null, // pending removal once the first paint has faded out
};

/* Whether the library has room to be a column: a screen narrower than a tablet
   gets it as a drawer over the document instead. Read from the window once and
   then kept up to date with it, because a window can be resized into the other
   shape while a reader is sitting in front of it. */
const NARROW = window.matchMedia("(max-width: 1023px)");
state.drawer = NARROW.matches;
NARROW.addEventListener("change", (event) => {
  state.drawer = event.matches;
  state.railOpen = false;
  renderRail();
});

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
  const text = value === undefined || value === null ? "" : String(value);
  /* Written only when it differs: replacing a column of text with itself is
     what throws a reader's scroll position away on every poll, and a long
     document pays for the rewrite even when nothing about it changed. */
  if (node.textContent !== text) {
    node.textContent = text;
  }
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

/* A notice is a passing message, not a permanent fixture: it leaves on its own
   before it can squat on the screen. A failure earns more reading time than a
   warning, and a plain confirmation the least. Clicking the bar waves it away
   at once, so a long error can still be read — or dismissed. */
const BANNER_MS = { error: 10000, warn: 6000, ok: 5000 };

/* How long the opacity transition takes. The bar is only made `hidden` (which
   is `display:none`, and cannot animate) after that, so it fades rather than
   blinking out. Reduced motion is handled by the stylesheet, which drops the
   transition and so makes the same wait invisible. */
const BANNER_FADE_MS = 400;

function clearBannerTimers() {
  clearTimeout(state.bannerTimer);
  clearTimeout(state.bannerFade);
  state.bannerTimer = null;
  state.bannerFade = null;
}

function renderBanner(kind, message) {
  const banner = byId("banner");
  if (!message) {
    hideBanner();
    return;
  }
  /* A new message cancels whatever the previous one was waiting to do, and
     takes back its exit: the reader is meant to see this one, not the last. */
  clearBannerTimers();
  banner.classList.remove("banner--leaving");
  banner.dataset.kind = kind;
  setText(banner, message);
  show(banner, true);
  state.bannerTimer = setTimeout(hideBanner, BANNER_MS[kind] || BANNER_MS.warn);
}

function hideBanner() {
  const banner = byId("banner");
  clearBannerTimers();
  if (banner.hidden) {
    setText(banner, ""); // never was shown: nothing to fade
    return;
  }
  banner.classList.add("banner--leaving");
  state.bannerFade = setTimeout(() => {
    banner.classList.remove("banner--leaving");
    show(banner, false);
    setText(banner, "");
    state.bannerFade = null;
  }, BANNER_FADE_MS);
}

/* True while the banner shows a content we wrote in this poll cycle (any
   kind): so a resume notice is not re-stamped on top of a banner the reader is
   already reading, and only the newest fact wins. */
function bannerActive() {
  const banner = byId("banner");
  return banner && !banner.hidden && banner.textContent !== "";
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
    /* The folded library shows one tile per document and a tile is a letter, so
       the row carries its own initial, its full name in the tooltip, and the
       words it is announced by — nothing is lost by folding the column. */
    const name = run.file || run.run_id;
    button.dataset.mark = (name.trim().charAt(0) || "?").toUpperCase();
    button.dataset.stateMark = STATE_MARKS[run.status] || "";
    button.title = name;
    const words = [name, STATUS_WORDS[run.status] || run.status];
    const when = formatTime(run.started);
    if (when) {
      words.push(when);
    }
    button.setAttribute("aria-label", words.join(" — "));
    button.append(element("span", "job-row-name", run.file || run.run_id));
    button.append(element("span", "job-row-time", formatTime(run.started)));
    const sub = element("span", "job-row-sub");
    /* The pair, the size and the state. */
    sub.append(element("span", null, `${run.src || "?"} → ${run.tgt || "?"}`));
    if (run.chunks) {
      sub.append(element("span", null, `${formatCount(run.chunks)} chunk`));
    }
    sub.append(element("span", null, STATUS_WORDS[run.status] || run.status));
    button.append(sub);
    button.addEventListener("click", () => {
      /* A row picked from the drawer is a choice made: the drawer has done its
         job and gives the document the whole window back. */
      closeRail();
      openJob(run.run_id);
    });
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

/* The mark a control gives way to when the pointer arrives: a pause for the
   button that ends a run, a play for the one that continues it. Static shapes,
   so they are built node by node like the trash glyph — the page never splices
   an HTML string. */
function markIcon(name) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "18");
  svg.setAttribute("height", "18");
  svg.setAttribute("fill", "currentColor");
  /* The mark repeats what the label already says, so it is decoration and is
     kept out of the accessible name rather than announced after it. */
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const shapes =
    name === "pause"
      ? [
          ["rect", { x: "6", y: "4.5", width: "4", height: "15", rx: "1.4" }],
          ["rect", { x: "14", y: "4.5", width: "4", height: "15", rx: "1.4" }],
        ]
      : [
          [
            "path",
            {
              d: "M8 5.14v13.72a1 1 0 0 0 1.53.85l10.4-6.86a1 1 0 0 0 0-1.7L9.53 4.29A1 1 0 0 0 8 5.14z",
            },
          ],
        ];
  for (const shape of shapes) {
    const node = document.createElementNS(NS, shape[0]);
    const attrs = shape[1];
    for (const key of Object.keys(attrs)) {
      node.setAttribute(key, attrs[key]);
    }
    svg.appendChild(node);
  }
  return svg;
}

/* A control whose word steps aside for its mark: the label slides out to the
   right and fades, the mark slides in from the left and takes its place. The
   word is still the button's name — it is the same action either way, and a
   reader who never hovers it (or a screen reader) still gets the sentence. */
function swapButton(label, mark, extraClass) {
  const button = element(
    "button",
    ["btn", "btn--swap", extraClass].filter(Boolean).join(" "),
  );
  button.type = "button";
  const word = element("span", "btn-label", label);
  const icon = element("span", "btn-mark");
  icon.append(markIcon(mark));
  button.append(word, icon);
  return button;
}

/*: Which mark each run control swaps to. */
const CONTROL_MARKS = { stop: "pause", resume: "play" };

function renderActions(detail) {
  const box = clear(byId("job-actions"));
  /* Starting and stopping are not downloads: "Dừng" and "Tiếp tục" live at the
     top-right of the panel, on the same line as the badge that says what the
     run is doing, so the control that changes that state is not two rows below
     the state it changes. */
  const control = clear(byId("job-control"));
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
  /* "Dừng" also covers a run waiting between two automatic attempts: it has no
     child to signal, and the server answers by calling the retry off, so the
     reader is never left watching a wait they cannot end. */
  const waiting = Number(run.retry_at) > 0;
  if (run.status === "running" || waiting) {
    const stop = swapButton("Dừng", CONTROL_MARKS.stop);
    stop.title = waiting ? "Huỷ lần tự động tiếp tục đang chờ" : "Kết thúc lượt dịch này";
    stop.addEventListener("click", () => stopRun(run.run_id));
    control.append(stop);
  }
  if (run.status === "failed" || run.status === "stopped") {
    /* "Tiếp tục" wears the warn tone the stopped run's own title and badge
       wear, so the button that undoes the stopping is the colour of it. */
    const resume = swapButton("Tiếp tục", CONTROL_MARKS.resume, "btn--resume");
    resume.addEventListener("click", () => resumeRun(run.run_id));
    control.append(resume);
  }
}

function renderJob(detail) {
  const run = detail.run;
  const job = byId("job");
  job.dataset.state = run.status;
  setText(byId("job-title"), run.file || run.run_id);
  /* The header names the document and says what it is: the pair of languages
     and where the run has got to. The name is the only part allowed to be
     long, so it is the only part that elides. */
  setText(byId("crumb-doc"), run.file || run.run_id);
  setText(
    byId("crumb-meta"),
    `${run.src || "?"} → ${run.tgt || "?"} · ${STATUS_WORDS[run.status] || run.status}`,
  );
  /* What the run was asked to do — the language pair, the chunk
     count, when it started — is not a headline under the title: those are facts
     about the translation, and they are listed with the rest of them in the
     report tab. The heading keeps the document's name and the badge that says
     what the run is doing. */
  const meta = clear(byId("job-meta"));
  show(meta, false);
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
  setText(byId("crumb-doc"), "Chưa mở lượt dịch nào");
  setText(byId("crumb-meta"), "");
  setText(
    byId("job-meta"),
    "Chọn một lượt ở cột bên trái, hoặc bắt đầu một lượt mới.",
  );
  show(byId("job-meta"), true);
  /* No job open means no state to name: an empty badge is a blob, not a word. */
  const badge = show(byId("job-status"), false);
  badge.dataset.state = "empty";
  setText(badge, "");
  clear(byId("job-actions"));
  clear(byId("job-control"));
  show(byId("job-alert"), false);
  show(byId("progress"), false);
  /* No run is open, so the detail returns to its first tab and the two lists
     have nothing to show. */
  state.tab = "result";
  renderTabs();
  renderChunks(null);
  resetLog();
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
  /* The downloads belonged to the run that was open: with no run there is no
     text behind them, and a link to the run that was just deleted would only
     fail. */
  renderSourceGets("", "empty");
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

/*: The rail's own short list: six phases rather than the pipeline's seventeen
   stage names. The pipeline reports on itself one stage at a time, and most of
   those stages run inside the per-chunk translation loop — `translate`,
   `critic`, `verify`, `refine` and the per-chunk `save_trace` interleave across
   chunks from the first one to the last — so a row per stage made a rail that
   was thirteen words long and flipped between "dịch" and "dịch lại" while the
   translation was still going. What a reader is waiting on is the phase, not
   the stage: the document is read and split, the plan is laid out, the input is
   translated and checked, and the result is made consistent, written and saved
   to experience. */
const STEP_GROUPS = [
  ["đọc & chia tài liệu", ["ingest", "detect_lang", "segment"]],
  ["lập kế hoạch", ["glossary", "summary_tree", "plan", "analyse_chunks"]],
  [
    "dịch & kiểm tra",
    ["translate", "critic", "refine", "verify", "retranslate", "collect", "save_trace"],
  ],
  ["nhất quán", ["consistency"]],
  ["ghép kết quả", ["assemble_output"]],
  ["lưu kinh nghiệm", ["skillopt_update"]],
];

/*: The pipeline's stage name → the phase it belongs to. A stage this map has
   never heard of gets a row of its own, under its own name, rather than being
   dropped out of the rail by a UI that has not been told about it. */
const STEP_PHASE = new Map();
for (const entry of STEP_GROUPS) {
  for (const stage of entry[1]) {
    STEP_PHASE.set(stage, entry[0]);
  }
}

/* Every phase the run has reached, in the order it reached them.

   A phase is ticked once the run has **moved past it** — the whole input has
   gone through it and the run is now reporting a later phase — or once the run
   is finished. A phase that runs once per chunk therefore stays "đang chạy" for
   as long as chunks are still coming, which is the honest answer: half a
   document translated is not a finished translation.

   A phase is marked failed only once the run has ended, which is when that
   verdict is final; what it stumbled on stays in its tooltip and in the log. */
function phaseStates(events, status) {
  const order = [];
  const seen = new Map();
  const finished = status === "done";
  const ended = isFinal(status);
  for (const event of events) {
    const stage = String(event.step || "");
    if (!stage) {
      continue;
    }
    const label = STEP_PHASE.get(stage) || stage;
    let entry = seen.get(label);
    if (!entry) {
      entry = { label: label, stages: new Set(), failed: 0, last: "ok" };
      seen.set(label, entry);
      order.push(label);
    }
    entry.stages.add(stage);
    entry.last = String(event.status || "ok");
    if (event.status === "fail") {
      entry.failed += 1;
    }
  }
  return order.map((label, index) => {
    const entry = seen.get(label);
    const newest = index === order.length - 1;
    let state = "running";
    if (finished || !newest) {
      state = "ok";
    } else if (ended && entry.last === "fail") {
      state = "fail";
    }
    return {
      label: label,
      state: state,
      stages: Array.from(entry.stages),
      failed: entry.failed,
    };
  });
}

function renderSteps(events, status) {
  const list = clear(byId("steps"));
  for (const phase of phaseStates(events, status)) {
    const item = element("li", "step", phase.label);
    item.dataset.state = phase.state;
    /* The row is short, so its tooltip is the exact one: the pipeline's own
       stage names behind the phase, what the phase is doing, and how often it
       stumbled. A row that turned red without saying any of that would have
       left the reader to guess. */
    item.title =
      `${phase.label} (${phase.stages.join(", ")}) — ` +
      `${STEP_STATE_WORDS[phase.state] || phase.state}` +
      (phase.failed ? ` (${formatCount(phase.failed)} lần lỗi trong lượt này)` : "");
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

/* ------------------------------------------------------------------ chunks */

/* The pipeline counts its chunks from zero — the first chunk of a document is
   `chunk-000`. A reader counts pages from one, and a list that starts at 000
   reads as an off-by-one bug rather than as a name. So the web UI renumbers
   for the reader: `chunk-001` is the first chunk, `chunk-002` the second, and
   the number is the chunk's place in the document (a 38-chunk run ends at
   `chunk-038`).

   The ids themselves are untouched: the log file, the checkpoints and the
   pipeline all still speak `chunk-000`. This is a display rule and nothing
   else. */
function displayChunkId(chunkId) {
  const id = String(chunkId || "");
  const match = id.match(/^(.*?)(\d+)$/);
  if (!match) {
    return id;
  }
  const digits = match[2];
  return `${match[1]}${String(Number(digits) + 1).padStart(digits.length, "0")}`;
}

/* The same renumbering inside a message: the pipeline writes its own errors
   with the raw id ("chunk-000: ..."), and a reader who sees `chunk-001` in the
   list must not find `chunk-000` in the message right next to it. */
function displayChunkText(text) {
  const value = String(text || "");
  if (!value) {
    return value;
  }
  return value.replace(
    /\bchunk-(\d+)\b/g,
    (_match, digits) => `chunk-${String(Number(digits) + 1).padStart(digits.length, "0")}`,
  );
}

/*: Each chunk state's colour and wording, so the reducer and the tooltip agree
   on the same four outcomes. */
const CHUNK_STATE_WORDS = {
  wait: "chờ",
  running: "đang chạy",
  ok: "xong",
  fail: "lỗi",
};

/*: What a chunk that stumbled earlier says instead of the step it stumbled on.
   The run does not abandon a chunk — it keeps translating the others while that
   one waits, and comes back to it — so the step it stopped at is history, not a
   verdict on the chunk. */
const CHUNK_LEFT_WORD = "lượt chạy đã đi tiếp";

/* Every chunk's state, derived from the events already polled. The pipeline
   names its chunks `chunk-000`, `chunk-001`, … in order and `run.chunks` says
   how many there are, so the grid shows the whole task — including chunks no
   event has mentioned yet, which read as "wait".

   The **last** event wins: a chunk's state is where it is now. Taking the
   strongest state instead would be wrong here — a chunk whose verification
   failed once, was retranslated and then saved has a `fail` in its past but is
   finished, and the whole run would read as broken when it is not. That hiccup
   is still visible: it stays in the tooltip and in the log tab. */
function chunkStates(events, total) {
  const byId = new Map();
  let highest = -1;
  for (const event of events) {
    const id = String(event.chunk_id || "");
    if (!id) {
      continue;
    }
    const status = String(event.status || "");
    const match = id.match(/(\d+)\s*$/);
    if (match) {
      highest = Math.max(highest, Number(match[1]));
    }
    let entry = byId.get(id);
    if (!entry) {
      entry = {
        id: id,
        index: match ? Number(match[1]) : null,
        state: "wait",
        step: "",
        error: "",
        recovered: false,
      };
      byId.set(id, entry);
    }
    if (status === "ok" || status === "patched") {
      entry.state = "ok";
    } else if (status === "fail") {
      entry.state = "fail";
    } else if (status === "start" || status === "retry") {
      entry.state = "running";
    }
    if (event.step) {
      entry.step = String(event.step);
    }
    const error = String(event.error || event.message || "");
    if (error) {
      entry.error = error;
    }
  }
  /* A failure is only a chunk's current state while the run is still on that
     chunk. The pipeline does not stop for one bad chunk: a step that hits a
     provider timeout holds that chunk back while the next ones are translated,
     so a `fail` on a chunk the run has already walked past is history, not a
     verdict — the list would otherwise show early chunks red while the chunks
     after them finish, which reads as a broken run that is plainly still
     working. `highest` is the furthest chunk the run has touched, so a chunk
     below it has been overtaken. The failure stays in the tooltip, in the
     message a row opens, in the "Chunk từng lỗi" filter and in the log. */
  for (const entry of byId.values()) {
    if (entry.state === "fail" && entry.index !== null && entry.index < highest) {
      entry.state = "ok";
      entry.recovered = true;
    }
  }
  const count = Math.max(Number(total) || 0, highest + 1);
  const cells = [];
  /* The chunker names its chunks from zero (`chunk-000`), so the cells are
     keyed the same way: they are the pipeline's own ids, the ones the events
     carry and the ones the log file holds. Only what the reader sees is
     renumbered (see `displayChunkId`), so counting from one cannot hide
     `chunk-000` or invent a chunk past the end, and the summary still agrees
     with the progress bar. */
  for (let i = 0; i < count; i += 1) {
    const id = `chunk-${String(i).padStart(3, "0")}`;
    cells.push(byId.get(id) || { id: id, state: "wait", step: "", error: "" });
  }
  return cells;
}

/* What one cell says on hover: which chunk, how far it got, and its problem. */
function chunkTitle(cell) {
  const parts = [displayChunkId(cell.id)];
  if (cell.step) {
    parts.push(STEP_WORDS[cell.step] || cell.step);
  }
  parts.push(CHUNK_STATE_WORDS[cell.state] || cell.state);
  if (cell.recovered) {
    parts.push(CHUNK_LEFT_WORD);
  }
  if (cell.error) {
    parts.push(
      `${cell.recovered ? "lỗi trước đó" : "lỗi"}: ${displayChunkText(cell.error)}`,
    );
  }
  return parts.join(" · ");
}

/* What a chunk is working on, in a word: the last step it reported, or nothing
   yet while it is still waiting its turn. A chunk the run left behind is not
   working on anything — naming the step it died on would read as an error
   beside a row that is finished. */
function chunkWork(cell) {
  if (cell.state === "wait" || !cell.step) {
    return "chưa bắt đầu";
  }
  if (cell.recovered) {
    return CHUNK_LEFT_WORD;
  }
  return STEP_WORDS[cell.step] || cell.step;
}

/* What the summary line says: the whole run's counts, and — when a filter is in
   force — what the list is actually showing. A filtered list that said nothing
   about itself would read as a run with almost no chunks in it.

   The `recovered` count is not an error count and not a lie either: it says how
   many of the finished chunks got there after a step of theirs failed, which is
   the thing a reader scanning a wall of green would otherwise never learn. */
function chunksSummary(counts, shown, total, recovered) {
  let words =
    `Xong ${formatCount(counts.ok)} · đang chạy ${formatCount(counts.running)}` +
    ` · lỗi ${formatCount(counts.fail)} · chờ ${formatCount(counts.wait)}`;
  if (recovered) {
    words += ` · ${formatCount(recovered)} chunk từng lỗi`;
  }
  if (state.chunkFilter !== "fail") {
    return words;
  }
  if (!total || shown) {
    return `${words} · đang lọc: chỉ chunk từng lỗi (${formatCount(shown)}/${formatCount(total)})`;
  }
  return `${words} · không có chunk nào từng lỗi`;
}

/* Opening one chunk's message closes the other: two open messages in a list this
   long would hide the rows between them. */
function toggleChunkError(chunkId) {
  state.chunkOpen = state.chunkOpen === chunkId ? null : chunkId;
  renderChunks(state.detail);
  /* Redrawing replaces the row that was just activated, which would drop a
     keyboard reader back to the top of the page. The same chunk's row takes the
     focus back, so Enter opens and Enter closes in place. */
  const list = byId("chunks");
  for (const row of list.children) {
    if (row.dataset.chunk === chunkId) {
      row.focus();
      return;
    }
  }
}

/* The whole task as a list, one row per chunk: its id, its state and the work
   it is doing. A reader scans down it to see where a long run stands and which
   chunks went wrong, and the list scrolls on its own so hundreds of rows do
   not push the page. */
function renderChunks(detail) {
  const list = byId("chunks");
  const summary = byId("chunks-summary");
  /* Rebuilding the rows empties the list, which drops its position to the top: a
     reader who had scrolled to chunk 60 of a hundred would be thrown back on
     every poll. Their place is given back before anything follows the run. */
  const scroll = list.scrollTop;
  clear(list);
  const cells = detail ? chunkStates(state.events, detail.run.chunks) : [];
  for (const entry of [
    ["chunk-filter-all", "all"],
    ["chunk-filter-fail", "fail"],
  ]) {
    byId(entry[0]).setAttribute(
      "aria-pressed",
      state.chunkFilter === entry[1] ? "true" : "false",
    );
  }
  if (!detail) {
    setText(summary, "");
    return;
  }
  const counts = { wait: 0, running: 0, ok: 0, fail: 0 };
  let recovered = 0;
  for (const cell of cells) {
    counts[cell.state] = (counts[cell.state] || 0) + 1;
    if (cell.recovered) {
      recovered += 1;
    }
  }
  /* The chunk the run is on, so the list can keep it in view. Chunks are
     translated side by side, so there can be more than one; the last is the
     frontier the run is moving down the document. */
  let running = null;
  let shown = 0;
  for (const cell of cells) {
    /* The filter answers "which chunks went wrong?", so it keeps a chunk that
       is failing now *and* one that failed and was left behind: those are
       exactly the chunks a reader opens this list to find, and a row that has
       gone green must not take its message out of reach with it. */
    if (state.chunkFilter === "fail" && cell.state !== "fail" && !cell.error) {
      continue;
    }
    shown += 1;
    const row = element("li", "chunk");
    row.dataset.state = cell.state;
    row.dataset.chunk = cell.id;
    if (cell.recovered) {
      row.dataset.recovered = "true";
    }
    row.title = chunkTitle(cell);
    row.append(element("span", "chunk-id", displayChunkId(cell.id)));
    row.append(element("span", "chunk-state", CHUNK_STATE_WORDS[cell.state] || cell.state));
    row.append(element("span", "chunk-work", chunkWork(cell)));
    /* A chunk with something to say can be opened to say all of it: the tooltip
       is where a reader glances, this is where a message is read and copied. A
       chunk with no message stays a plain row rather than becoming a control
       that opens onto nothing. */
    if (cell.error) {
      const open = state.chunkOpen === cell.id;
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-expanded", open ? "true" : "false");
      row.addEventListener("click", () => toggleChunkError(cell.id));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggleChunkError(cell.id);
        }
      });
      if (open) {
        row.append(element("span", "chunk-error", displayChunkText(cell.error)));
      }
    }
    list.append(row);
    if (cell.state === "running") {
      running = row;
    }
  }
  setText(summary, chunksSummary(counts, shown, cells.length, recovered));
  list.scrollTop = scroll;
  /* The status is watched, not read once: a run of a hundred chunks moves out
     of sight within a few steps. `nearest` moves the list only when the chunk
     has actually left it, so following costs nothing while it is in view. */
  if (running && state.follow) {
    running.scrollIntoView({ block: "nearest" });
  }
}

/* Whether the chunk the run is on sits inside the visible part of the list.
   Reading the reader's own scroll is what lets following stop by itself: they
   scrolled somewhere else, so the run no longer leads the page — and scrolling
   back to it starts following again, with no control to explain. */
function followRunning() {
  const list = byId("chunks");
  const rows = list.querySelectorAll('.chunk[data-state="running"]');
  if (!rows.length) {
    return state.follow; // nothing is running: the reader's choice stands
  }
  const listBox = list.getBoundingClientRect();
  const rowBox = rows[rows.length - 1].getBoundingClientRect();
  return rowBox.top < listBox.bottom && rowBox.bottom > listBox.top;
}

/* The pipeline's own early `resume` record: how many chunks this attempt
   replayed from checkpoint instead of re-translating, and whether the store had
   any matching chunk to replay at all.

   Every attempt writes one, a fresh run included — the record reports on the
   store, and the store exists before the run knows which it is. So
   `resume_requested` is what separates the two, and a fresh run's record says
   nothing a reader needs. */
function resumeInfo(events) {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (event && event.type === "resume") {
      return {
        requested: Boolean(event.resume_requested),
        resumed: Number(event.resumed_chunks) || 0,
        usable: Boolean(event.checkpoint_usable),
        /* When this attempt was continued. The notice belongs to that moment,
           so a page can tell a notice it has already raised from one it has
           not. */
        ts: Number(event.ts) || 0,
      };
    }
  }
  return null;
}

/* When the attempt the reader is watching began.

   A run's own start time is the day the run was created, and a run that was
   continued keeps it: the time it spent failed — waiting for a reader, for a
   provider, for a free slot — is not time it spent translating. Counting from
   the run's start made a run that had just been continued report the whole
   wait as running time, and a healthy run read as one that had been stuck for
   an hour. The clock measures the attempt, so it starts again with every
   attempt; when the log has no attempt to point at, the run's start is the
   best answer there is. */
function attemptStart(events, run) {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event && event.type === "run_start" && event.ts) {
      return Number(event.ts);
    }
  }
  return Number(run && run.started) || 0;
}

/* What to say about a resume that replayed nothing. `checkpoint_usable` only
   means checkpointing is switched on — it is not evidence that a model or file
   changed. A previous attempt that was stopped before finishing a single chunk
   produces the exact same reading, so claiming a mismatch every time is wrong.
   Say only what is certain: no checkpoint was reused. */
function resumeNotice(info) {
  return info.usable
    ? "Không dùng lại được checkpoint nào cho lượt này — các chunk sẽ dịch từ đầu."
    : "Chưa có checkpoint để dùng — các chunk sẽ dịch từ đầu.";
}

/* How far back the token rate looks. A window, not the run's average: an
   average survives a stall — five good minutes hide a two-minute one — while a
   short window drops to zero within seconds of the provider going quiet, which
   is exactly when a reader needs to see it. */
const SPEED_WINDOW_S = 30;

/* Tokens per second over the last SPEED_WINDOW_S, from the step_end events the
   page has already polled: input + output, summed over the window. "-" until
   the run has produced a token at all ("not yet" must not read as "zero"), a
   plain `0 token/s` when the window is empty — the stall itself is the news,
   no annotation — and gone entirely once the run is over, because a finished
   run's numbers belong to the report tab. */
function speedLine(run) {
  if (run.status !== "running") return "";
  const now = Date.now() / 1000;
  const started = Number(run.started) || now;
  const span = Math.min(SPEED_WINDOW_S, Math.max(0, now - started));
  if (span < 1) return "";
  const cutoff = now - span;
  let sum = 0;
  let ever = false;
  for (let i = state.events.length - 1; i >= 0; i -= 1) {
    const tokens = state.events[i].tokens;
    if (!tokens) continue;
    const produced = Number(tokens.sent || 0) + Number(tokens.received || 0);
    if (produced > 0) ever = true;
    if (Number(state.events[i].ts) >= cutoff) sum += produced;
  }
  if (!ever) return "\u2014";
  if (sum <= 0) return "0 token/s";
  return `≈ ${formatCount(Math.round(sum / span))} token/s`;
}

function renderProgress(detail) {
  const run = detail.run;
  const box = byId("progress");
  show(box, true);

  const bar = byId("bar");
  const fill = byId("bar-fill");
  const total = Number(run.chunks) || 0;
  const eventDone = chunksDone(state.events);
  /* What the run replayed from checkpoint, and what its own events show, are
     two witnesses of the same work once "Tiếp tục" continues a run in place:
     the replayed chunks have their events in this very log. Adding the two
     would count them twice — 34 chunks finished before a stop would read as
     68/N. Take the larger of the two: at least the events show, and at least
     the store replayed. */
  const replayed = (resumeInfo(state.events) || {}).resumed || 0;
  const done = Math.min(total, Math.max(eventDone, replayed));
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

  const info = resumeInfo(state.events);
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
  const began = attemptStart(state.events, run);
  setText(
    byId("progress-elapsed"),
    until && began ? `đã chạy ${formatClock(until - began)}` : "",
  );

  setText(byId("progress-speed"), speedLine(run));
  setText(byId("progress-step"), progressWord(run, last, info));
  /* The rail is told how the run ended: a tick means the whole pass is done. */
  renderSteps(state.events, run.status);
  /* Each panel is drawn only while it is the visible tab: a run of a few
     hundred chunks, or a log of a few thousand lines, does not need its nodes
     re-made for a panel nobody is looking at. */
  if (state.tab === "status") {
    renderChunks(detail);
  } else if (state.tab === "history") {
    renderLog();
  } else if (state.tab === "report") {
    renderReport(detail);
  }
}

function progressWord(run, last, info) {
  if (run.status === "running") {
    /* A resumed run says what it is doing like any other run: the resume is a
       line in the log, not a status. Holding this caption on "đang resume" for
       the rest of the run is what left a reader watching one frozen sentence
       while chunks were still being translated. */
    const step = last ? String(last.step || "") : "";
    const where = step ? STEP_WORDS[step] || step : "";
    /* A chunk that failed and is waiting to be run again is still working. A
       run that says nothing while it waits out a provider outage reads as a
       hang, so the wait says what it is and how long it is. */
    if (last && String(last.status || "") === "retry") {
      const why = displayChunkText(String(last.error || last.message || ""));
      const on = last.chunk_id
        ? `${where || "chunk"} · ${displayChunkId(last.chunk_id)}`
        : where || "chunk";
      /* A slot collision is the provider holding a request we already gave up
         on — a wait for *them*, not our own retry: name it as such. */
      const slot = /concurrency_limit_exceeded|holds the slot|concurrent request/i.test(why);
      const head = slot ? "Đang chờ provider nhả slot" : "Đang chờ thử lại";
      return `${head}: ${on}${why ? ` — ${why}` : ""}`;
    }
    if (step) {
      return `Đang chạy: ${
        last.chunk_id ? `${where} · ${displayChunkId(last.chunk_id)}` : where
      }`;
    }
    /* Before this attempt's first step, a resume whose store had no matching
       chunk to replay says it re-translates rather than leaving the reader to
       watch the bar start over with no explanation. */
    if (info && info.requested && info.resumed === 0) {
      return resumeNotice(info);
    }
    return "Đang chuẩn bị…";
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

/* ---------------------------------------------------------------- history */

/*: One word per event status, so a log line reads as a sentence rather than as
   the pipeline's own JSON. A status added later shows as itself, not as a blank. */
const LOG_STATUS_WORDS = {
  start: "bắt đầu",
  ok: "xong",
  fail: "lỗi",
  retry: "thử lại",
  patched: "đã vá",
  info: "ghi chú",
  note: "tiếp tục",
};

/* When a log line happened, to the second: a log is read in order, and the gaps
   between its lines are part of what it says. A line the pipeline itself never
   stamped — the server's note about a run that died silently — has no time. */
function logClock(epochSeconds) {
  const seconds = Number(epochSeconds) || 0;
  if (seconds <= 0) {
    return "";
  }
  const when = new Date(seconds * 1000);
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(when.getHours())}:${pad(when.getMinutes())}:${pad(when.getSeconds())}`;
}

/* One line of the run's log, or null for a record that is not work.

   A `decision` is routing detail — which route a chunk was sent down — and it
   is already summarised in "Đường đi"; two lines per chunk would bury the work
   they were deciding about. */
function logEntry(event) {
  const kind = String(event.type || "");
  /* Two records are not work. A `decision` is routing detail already summarised
     in "Đường đi"; a `checkpoint` is the pointer the page reads to find the
     chunks a run has written. Neither has anything to say in the log, and an
     empty row is not a record of anything. */
  if (kind === "decision" || kind === "checkpoint") {
    return null;
  }
  const status = String(event.status || "ok");
  const step = String(event.step || "");
  const cause = String(event.cause || "");
  const message = String(event.error || event.message || "");
  const entry = {
    ts: event.ts,
    status: status,
    where: STEP_WORDS[step] || step,
    chunk: displayChunkId(event.chunk_id),
    text: displayChunkText(
      [CAUSE_WORDS[cause] || cause, message].filter(Boolean).join(" · "),
    ),
    /* The event itself, so a row can be opened onto everything it carries —
       the full message and the fields the one line leaves out. */
    event: event,
  };
  if (kind === "run_start") {
    entry.status = "ok";
    entry.where = "Bắt đầu";
    entry.text = [
      event.file ? String(event.file) : "",
      event.chunks ? `${formatCount(event.chunks)} chunk` : "",
      event.src || event.tgt ? `${event.src || "?"} → ${event.tgt || "?"}` : "",
    ]
      .filter(Boolean)
      .join(" · ");
  } else if (kind === "run_end") {
    entry.status = status === "failed" ? "fail" : "ok";
    entry.where = "Kết thúc";
    const calls = Number(event.active_api_calls);
    const peak = Number(event.max_active_api_calls);
    const callWords = [];
    if (Number.isFinite(calls) && calls >= 0) {
      callWords.push(`${formatCount(calls)} luồng`);
    }
    if (Number.isFinite(peak) && peak > 0) {
      callWords.push(`cao nhất ${formatCount(peak)}`);
    }
    entry.text = [
      event.chunks
        ? `đã dịch ${formatCount(event.chunks_done || 0)}/${formatCount(event.chunks)} chunk`
        : "",
      event.chunks_failed ? `${formatCount(event.chunks_failed)} chunk lỗi` : "",
      callWords.length ? callWords.join(" · ") : "",
      event.duration_ms ? formatClock(Number(event.duration_ms) / 1000) : "",
    ]
      .filter(Boolean)
      .join(" · ");
  } else if (kind === "resume") {
    /* Only the log carries this. Which chunks a resume replayed says what the
       run's past was, not what it is doing now, so it is not pinned under the
       progress bar for the rest of the run — the reader comes here for it. */
    if (!event.resume_requested) {
      /* A fresh attempt writes this record too, and drawing it put "Tiếp tục ·
         Không dùng lại được checkpoint nào cho lượt này — các chunk sẽ dịch từ
         đầu" at the top of a run that had never been resumed and had no past to
         reuse: it read as a resume that had thrown everything away. Nothing
         happened, so there is no line to print. */
      return null;
    }
    entry.status = "note";
    entry.where = "Tiếp tục";
    entry.text = Number(event.resumed_chunks)
      ? `dùng lại ${formatCount(event.resumed_chunks)} chunk đã dịch trước đó`
      : resumeNotice({ usable: Boolean(event.checkpoint_usable) });
  } else if (!entry.text && (event.duration_ms || event.latency)) {
    entry.text = formatDuration(Number(event.duration_ms || event.latency));
  }
  return entry;
}

/* The chevron a log line folds behind: a static shape, built node by node like
   the other glyphs — the page never splices an HTML string. */
function chevronIcon() {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "12");
  svg.setAttribute("height", "12");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2.5");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(NS, "path");
  path.setAttribute("d", "M9 6l6 6-6 6");
  svg.appendChild(path);
  return svg;
}

/* What one event carries that its single line does not say: the fields with a
   value, each under its own name. Keys the line already speaks — the clock, the
   step, the chunk, the verdict, the message — stay out, and a key this page has
   never heard of still shows under its own name rather than disappearing. */
function logDetailRows(event) {
  const shown = new Set(["ts", "run_id", "step", "chunk_id", "status", "error", "message"]);
  const rows = [];
  for (const key of Object.keys(event)) {
    if (shown.has(key)) {
      continue;
    }
    const value = event[key];
    if (value === "" || value === null || value === undefined || value === false) {
      continue;
    }
    if (typeof value === "number" && value === 0) {
      continue;
    }
    let text;
    if (key === "latency" || key === "duration_ms") {
      text = formatDuration(Number(value));
    } else if (key === "tokens" && typeof value === "object") {
      text = Object.entries(value)
        .filter(([, n]) => Number(n))
        .map(([k, n]) => `${k}: ${formatCount(Number(n))}`)
        .join(" · ");
    } else if (typeof value === "object") {
      text = JSON.stringify(value);
    } else {
      text = String(value);
    }
    if (!text) {
      continue;
    }
    rows.push([LOG_DETAIL_WORDS[key] || key, text]);
  }
  return rows;
}

/* One line of the run's log, openable: the fold holds the message to two lines
   and the event's own fields stay behind the chevron, so a log of a thousand
   steps remains scannable and a long provider error is still one click away. */
function logRow(entry) {
  const row = element("li", "log-line");
  const word = LOG_STATUS_WORDS[entry.status] || entry.status;
  row.dataset.status = entry.status;
  row.dataset.expanded = "false";
  row.title = [entry.where, entry.chunk, word, entry.text].filter(Boolean).join(" · ");
  const fields = entry.event ? logDetailRows(entry.event) : [];
  /* A line with nothing to open — no message, no fields — stays a plain row
     rather than becoming a control that opens onto nothing. */
  const toggle = element("button", "log-toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-label", `Xem chi tiết dòng lịch sử: ${entry.where || word}`);
  toggle.append(chevronIcon());
  if (!entry.text && !fields.length) {
    toggle.disabled = true;
    toggle.setAttribute("aria-hidden", "true");
    toggle.tabIndex = -1;
  }
  row.append(toggle);
  row.append(element("span", "log-time", logClock(entry.ts)));
  row.append(
    element(
      "span",
      "log-where",
      entry.chunk ? `${entry.where} · ${entry.chunk}` : entry.where || "—",
    ),
  );
  row.append(element("span", "log-word", word));
  row.append(element("span", "log-text", entry.text));
  const detail = element("div", "log-detail");
  detail.hidden = true;
  for (const pair of fields) {
    const line = element("p", "log-detail-line");
    line.append(element("span", "log-detail-key", pair[0]));
    line.append(element("span", "log-detail-value", pair[1]));
    detail.append(line);
  }
  row.append(detail);
  /* Rows the server notes itself (a child that died before its first event)
     carry no event: the fold still opens the message, there are just no fields
     to show under it. */
  const open = () => {
    const isOpen = row.dataset.expanded !== "true";
    row.dataset.expanded = isOpen ? "true" : "false";
    toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    detail.hidden = !isOpen;
  };
  toggle.addEventListener("click", open);
  return row;
}

/* The problems the log already carries as lines of its own: the server derives
   its list from this very feed, so matching a problem back to a line keeps one
   event from being printed twice. */
function loggedProblemKeys() {
  const keys = new Set();
  for (const event of state.events) {
    const status = String(event.status || "");
    if (status === "fail" || status === "retry") {
      keys.add(`${event.step || ""}::${event.chunk_id || ""}`);
    }
  }
  return keys;
}

function resetLog() {
  state.logged = 0;
  state.loggedProblems = new Set();
  clear(byId("history"));
}

/* The run's log, one line per event, appended a poll at a time: what each chunk
   is doing, and — because the events are kept in order — how a chunk that
   stumbled went on to finish, instead of the stumble standing as its verdict
   for the rest of the run. */
function renderLog() {
  const list = byId("history");
  const atEnd = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  for (let i = state.logged; i < state.events.length; i += 1) {
    const entry = logEntry(state.events[i]);
    state.logged = i + 1;
    if (entry) {
      list.append(logRow(entry));
    }
  }
  /* The child's own last words, when it died before writing any event: the
     server reports them and nothing in the feed carries them. */
  const covered = loggedProblemKeys();
  for (const problem of (state.detail && state.detail.problems) || []) {
    const key = `${problem.step || ""}::${problem.chunk_id || ""}`;
    if (covered.has(key) || state.loggedProblems.has(key)) {
      continue;
    }
    state.loggedProblems.add(key);
    const cause = String(problem.cause || "");
    list.append(
      logRow({
        ts: 0,
        status: "fail",
        where: STEP_WORDS[problem.step] || problem.step || "Lượt chạy",
        chunk: displayChunkId(problem.chunk_id),
        text: displayChunkText(
          [CAUSE_WORDS[cause] || cause, String(problem.message || "")]
            .filter(Boolean)
            .join(" · "),
        ),
      }),
    );
  }
  /* Only a reader already at the end wants to be carried along; someone
     scrolling back through the log is left where they are. */
  if (atEnd) {
    list.scrollTop = list.scrollHeight;
  }
}

/* ----------------------------------------------------------------- report */

/* A step's own timing, in the unit a reader thinks in. The pipeline counts
   milliseconds — `duration_ms` on a step's end, and `latency` in the same
   unit — so a step that took six seconds arrives as 6000; printing that number
   beside an "s" would claim a nine-minute run took nine hours. */
function formatDuration(ms) {
  const value = Math.max(0, Number(ms) || 0);
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  const seconds = value / 1000;
  return seconds < 60 ? `${seconds.toFixed(1).replace(".", ",")} s` : formatClock(seconds);
}

function lastOfType(events, type) {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (events[i] && events[i].type === type) {
      return events[i];
    }
  }
  return null;
}

/* What the events already say about each chunk: what the pipeline spent on it,
   and how many times it had to translate it again. */
function chunkTotals(events) {
  const byChunk = new Map();
  for (const event of events) {
    const id = String(event.chunk_id || "");
    if (!id) {
      continue;
    }
    let entry = byChunk.get(id);
    if (!entry) {
      entry = { id: id, ms: 0, retranslated: 0 };
      byChunk.set(id, entry);
    }
    entry.ms += Number(event.duration_ms || event.latency || 0);
    if (String(event.step || "") === "retranslate" && String(event.status) === "ok") {
      entry.retranslated += 1;
    }
  }
  return byChunk;
}

/* The end-of-run report, built from the events the page already polls: no route
   of its own, and a run that is still going gets working figures rather than a
   blank page. `run_end` is the pipeline's own tally, so it wins where it
   exists; everything per chunk is counted here. */
function reportFor(detail) {
  const run = detail.run;
  const events = state.events;
  const cells = chunkStates(events, run.chunks);
  const counts = { wait: 0, running: 0, ok: 0, fail: 0 };
  for (const cell of cells) {
    counts[cell.state] += 1;
  }
  const totals = Array.from(chunkTotals(events).values());
  const retried = totals.filter((entry) => entry.retranslated > 0);
  const end = lastOfType(events, "run_end");
  const last = events.length ? Number(events[events.length - 1].ts) || 0 : 0;
  /* The attempt's start, not the run's: a run that was continued keeps the day
     it was created, and the time it spent failed is not time it spent
     translating. */
  const started = attemptStart(events, run);
  /* Actual in-flight API calls at the newest observation, plus the peak seen in
     this run. Older events without the field leave both null: the UI says
     "not measured" instead of inventing a zero. */
  let activeApiCalls = null;
  let maxActiveApiCalls = null;
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    /* `Number(null)` is 0, and most step events carry these fields as null —
       an event that simply does not know must not be read as "no calls in
       flight", or the scan stops there and the peak stays at zero. */
    const active =
      event.active_api_calls == null ? NaN : Number(event.active_api_calls);
    const peak =
      event.max_active_api_calls == null
        ? NaN
        : Number(event.max_active_api_calls);
    if (activeApiCalls === null && Number.isFinite(active) && active >= 0) {
      activeApiCalls = active;
    }
    if (Number.isFinite(peak) && peak >= 0) {
      maxActiveApiCalls = maxActiveApiCalls === null ? peak : Math.max(maxActiveApiCalls, peak);
    }
    if (activeApiCalls !== null && maxActiveApiCalls !== null) {
      break;
    }
  }
  return {
    run: run,
    end: end,
    finished: isFinal(run.status),
    counts: counts,
    failed: cells.filter((cell) => cell.state === "fail"),
    retried: retried,
    retranslateRuns: retried.reduce((sum, entry) => sum + entry.retranslated, 0),
    activeApiCalls: activeApiCalls,
    maxActiveApiCalls: maxActiveApiCalls,
    /* Only the slowest few: a table of every chunk's timing is the chunk list
       again, and no one reads it. */
    slowest: totals
      .filter((entry) => entry.ms > 0)
      .sort((a, b) => b.ms - a.ms)
      .slice(0, 8),
    resumed: (resumeInfo(events) || {}).resumed || 0,
    elapsed:
      end && end.duration_ms
        ? Number(end.duration_ms)
        : started && last
          ? Math.max(0, (last - started) * 1000)
          : 0,
  };
}

/* The consistency pass's tally on the report page.

   `run_end` carries four numbers now: terms that drifted between chunks, chunks
   rewritten to the planned term, chunks the budget left drifting, and rewrites
   the pass refused because they would have cost the translation. The last two
   were only ever visible as lines buried in the log, so a run that left most of
   its drift in the file read like one that found nothing to fix. A run with no
   drift has nothing to say here, and gets no line. */
function consistencyText(end) {
  if (!end) {
    return "";
  }
  const drifts = Number(end.consistency_inconsistencies) || 0;
  const patched = Number(end.consistency_patched) || 0;
  const skipped = Number(end.consistency_skipped) || 0;
  const rejected = Number(end.consistency_rejected) || 0;
  if (!drifts && !patched && !skipped && !rejected) {
    return "";
  }
  return [
    drifts ? `${formatCount(drifts)} thuật ngữ lệch` : "",
    patched ? `${formatCount(patched)} chunk đã vá` : "",
    skipped ? `${formatCount(skipped)} chunk bỏ sót` : "",
    rejected ? `${formatCount(rejected)} bản vá bị từ chối` : "",
  ]
    .filter(Boolean)
    .join(" · ");
}

function reportFacts(report) {
  const { counts, run, end, activeApiCalls, maxActiveApiCalls } = report;
  const total = counts.wait + counts.running + counts.ok + counts.fail;
  const facts = [];
  facts.push([
    "Chunk",
    `${formatCount(counts.ok)}/${formatCount(total)} xong` +
      (counts.fail ? ` · ${formatCount(counts.fail)} lỗi` : "") +
      (counts.wait ? ` · ${formatCount(counts.wait)} chưa dịch` : "") +
      (counts.running ? ` · ${formatCount(counts.running)} đang dịch` : ""),
  ]);
  /* A finished run has no "currently": its last sample is always zero, so the
     peak is the only number that means anything and the answer says so. */
  facts.push([
    "Api calls",
    activeApiCalls === null && maxActiveApiCalls === null
      ? "chưa đo được"
      : report.finished
        ? "lượt đã kết thúc" +
          (maxActiveApiCalls !== null
            ? ` · cao nhất ${formatCount(maxActiveApiCalls)} luồng`
            : "")
        : `${activeApiCalls === null ? "—" : formatCount(activeApiCalls)} luồng` +
          (maxActiveApiCalls !== null
            ? ` · cao nhất ${formatCount(maxActiveApiCalls)} luồng`
            : ""),
  ]);
  if (report.elapsed) {
    facts.push(["Thời gian chạy", formatClock(report.elapsed / 1000)]);
  }
  facts.push([
    "Phải dịch lại",
    report.retried.length
      ? `${formatCount(report.retranslateRuns)} lần ở ${formatCount(report.retried.length)} chunk`
      : "không có",
  ]);
  if (report.resumed) {
    facts.push(["Dùng lại checkpoint", `${formatCount(report.resumed)} chunk`]);
  }
  const consistency = consistencyText(end);
  if (consistency) {
    facts.push(["Nhất quán thuật ngữ", consistency]);
  }
  if (end && Number(end.retries)) {
    facts.push(["Lượt thử lại của nhà cung cấp", formatCount(end.retries)]);
  }
  const tokens = (end && end.tokens) || null;
  if (tokens && (tokens.sent || tokens.received)) {
    facts.push([
      "Token",
      `Input: ${formatCount(tokens.sent || 0)} tokens - Output: ${formatCount(tokens.received || 0)} tokens` +
        (tokens.cached ? ` - Cached input ${formatCount(tokens.cached)} tokens` : ""),
    ]);
  }
  /* What this run was asked to do: the language pair, how the
     document was split and when it started. These used to sit in a row of tags
     under the title; they belong with the rest of the run's figures. */
  facts.push(["Cặp ngôn ngữ", `${run.src || "?"} → ${run.tgt || "?"}`]);
  if (run.chunks) {
    facts.push(["Số chunk", formatCount(run.chunks)]);
  }
  if (run.started) {
    facts.push(["Bắt đầu", formatTime(run.started)]);
  }
  facts.push([
    "Kết thúc",
    end
      ? Number(end.exit_code) === 0
        ? "thành công"
        : `lỗi (mã thoát ${end.exit_code})`
      : STATUS_WORDS[run.status] || run.status,
  ]);
  return facts;
}

function reportTable(title, headings, rows, empty) {
  const box = element("section", "report-block");
  box.append(element("h3", "card-title", title));
  if (!rows.length) {
    box.append(element("p", "note", empty));
    return box;
  }
  const table = element("table", "report-table");
  const head = element("tr");
  for (const heading of headings) {
    head.append(element("th", "", heading));
  }
  table.append(head);
  for (const cells of rows) {
    const row = element("tr");
    for (const cell of cells) {
      row.append(element("td", "", cell));
    }
    table.append(row);
  }
  box.append(table);
  return box;
}

function renderReport(detail) {
  const box = clear(byId("report"));
  if (!detail) {
    return;
  }
  const report = reportFor(detail);
  if (!report.finished) {
    /* A run that is still going has working figures, not a verdict: say which
       it is rather than letting a reader file it away as final. */
    box.append(
      element("p", "note", "Lượt đang chạy — số liệu tạm tính, sẽ đầy đủ khi lượt kết thúc."),
    );
  }
  const facts = element("section", "report-facts");
  for (const entry of reportFacts(report)) {
    const row = element("div", "fact");
    row.append(element("span", "fact-label", entry[0]));
    row.append(element("span", "fact-value", entry[1]));
    facts.append(row);
  }
  box.append(facts);
  box.append(
    reportTable(
      "Chunk lỗi",
      ["Chunk", "Bước", "Lỗi"],
      report.failed.map((cell) => [
        displayChunkId(cell.id),
        STEP_WORDS[cell.step] || cell.step || "—",
        displayChunkText(cell.error || "(không có mô tả)"),
      ]),
      "Không có chunk nào lỗi.",
    ),
  );
  box.append(
    reportTable(
      "Chunk chậm nhất",
      ["Chunk", "Thời gian", "Dịch lại"],
      report.slowest.map((entry) => [
        displayChunkId(entry.id),
        formatDuration(entry.ms),
        entry.retranslated ? `${formatCount(entry.retranslated)} lần` : "—",
      ]),
      "Chưa có bước nào chạy xong để đo thời gian.",
    ),
  );
  box.append(
    reportTable(
      "Chunk phải dịch lại",
      ["Chunk", "Số lần"],
      report.retried
        .sort((a, b) => b.retranslated - a.retranslated)
        .map((entry) => [displayChunkId(entry.id), `${formatCount(entry.retranslated)} lần`]),
      "Không chunk nào phải dịch lại.",
    ),
  );
}

/* ------------------------------------------------------------------ panes */

/* The pipeline carries typography across the model as counted sentinels —
   ⟦BOLD⟧…⟦/BOLD⟧, ⟦ITALIC⟧…⟦/ITALIC⟧, ⟦CODE_n⟧ — machinery, not content: the
   reader asked for the words and the heading marks, never the type tags.
   Stripped on the way into a pane, so a run in flight (partial text) and a
   finished artefact read the same. */
const TYPE_MARKER_RE = /⟦(?:[/]?(?:BOLD|ITALIC)|CODE_[0-9]+)⟧/g;

function stripTypeMarkers(text) {
  return text ? String(text).replace(TYPE_MARKER_RE, "") : text;
}

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
  const text = byId(textId);
  /* A column filling in behind a running translation is redrawn on every chunk
     that lands, and drawing it from the top would throw the reader's place
     away each time. Only an append keeps the scroll — the text a reader was
     reading is still there, with more under it — while a different document
     starts at the top, which is where its text begins. */
  const shown = stripTypeMarkers(body);
  const kept = text.textContent && shown.startsWith(text.textContent) ? text.scrollTop : 0;
  setText(text, shown);
  if (kept) {
    text.scrollTop = kept;
  }
  setText(byId(metaId), meta);
}

/* The two files the source column can give: the extraction as the pipeline
   wrote it — ⟦CODE_n⟧ and the emphasis tokens intact — and the same text with
   the emphasis written as Markdown and the code markers gone. Built once per
   run: a poll redraws the panes every second, and rebuilding two links under a
   reader's focus would take it away for nothing. */
function renderSourceGets(base, sourceState) {
  const box = byId("source-gets");
  show(box, sourceState === "done");
  if (sourceState !== "done" || box.dataset.base === base) {
    return;
  }
  clear(box);
  for (const item of [
    {
      label: "Tải bản trích xuất",
      title: "Bản gốc trích xuất, giữ nguyên marker (⟦CODE_n⟧, ⟦BOLD⟧, ⟦ITALIC⟧)",
      markers: "keep",
    },
    {
      label: "Tải bản gốc",
      title: "Tải văn bản phía dưới",
      markers: "strip",
    },
  ]) {
    const link = element("a", "", item.label);
    link.href = `${base}/source?download=1&markers=${item.markers}`;
    link.title = item.title;
    link.setAttribute("download", "");
    box.append(link);
  }
  box.dataset.base = base;
}

function renderPanes() {
  const detail = state.detail;
  if (!detail) {
    renderSourceGets("", "empty");
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
  renderSourceGets(base, source.state);

  const target = { state: "empty", note: "", body: "", meta: "" };
  if (!detail.md) {
    if (run.status === "running") {
      /* The chunks the run has already finished are on disk from the moment
         each one lands, so the column shows the translation as it arrives
         instead of an empty page promising something later. What it shows is
         always a prefix of the final text — same chunks, same order, more of
         them — and before the first chunk there is nothing but the wait, said
         plainly. Nothing here is guessed: the pieces are the run's own, and
         the count is the number of them that exist. */
      if (state.partialText) {
        target.state = "done";
        target.body = state.partialText;
        target.note =
          `Bản dịch đang chạy — ${formatCount(state.partialPieces)} chunk đã dịch, phần còn lại sẽ nối tiếp vào đây.`;
      } else {
        target.state = "loading";
        target.note = "Bản dịch đang được tạo — nội dung sẽ hiện ở đây khi lượt chạy ghi ra file.";
      }
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

/* The column a reader compares against fills in behind the run: every chunk it
   has finished is written to its own file, and the run's own events count how
   many that is. So the text is asked for only when that count moves — watching
   a hundred chunks costs a hundred small reads rather than one per poll — and a
   look that fails leaves the last text standing in silence, because a
   convenience is not worth a banner over the run it belongs to. */
async function refreshPartial() {
  if (state.detail && state.detail.md) {
    return; // the whole translation is written: there is nothing partial to show
  }
  const done = chunksDone(state.events);
  if (done === state.partialDone) {
    return;
  }
  state.partialDone = done;
  try {
    const payload = await api(
      `/api/runs/${encodeURIComponent(state.open)}/partial`,
    );
    if (state.panesFor !== currentRunId()) {
      return; // the reader moved to another run
    }
    state.partialText = payload.text || "";
    state.partialPieces = Number(payload.pieces) || 0;
  } catch (error) {
    state.partialDone = -1; // whatever went wrong, the next poll may look again
  }
  renderPanes();
}

/* Which detail tab is open: the panes ("result"), the chunk list ("status"),
   the log ("history") or the end-of-run report ("report"). Every panel stays in
   the page; only visibility and the `aria-selected` state move. */
function renderTabs() {
  for (const entry of [
    ["tab-result", "result"],
    ["tab-status", "status"],
    ["tab-history", "history"],
    ["tab-report", "report"],
  ]) {
    byId(entry[0]).setAttribute(
      "aria-selected",
      entry[1] === state.tab ? "true" : "false",
    );
  }
  for (const name of ["result", "status", "history", "report"]) {
    show(byId(`tab-panel-${name}`), state.tab === name);
  }
}

function renderView() {
  const panes = byId("panes");
  panes.dataset.view = state.view;
  panes.dataset.zoom = state.zoom ? "on" : "off";
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
  byId("view-sync").setAttribute("aria-pressed", state.sync ? "true" : "false");
  /* The widened workspace is a toggle, so it says which way it goes next, in
     the tooltip and in the name a screen reader reads out. */
  const zoom = byId("view-zoom");
  zoom.setAttribute("aria-pressed", state.zoom ? "true" : "false");
  const word = state.zoom ? "Thu lại vùng đọc" : "Mở rộng vùng đọc";
  zoom.setAttribute("aria-label", word);
  zoom.title = word;
  renderPanes();
}

/* Which layout is in force, and where each column was left. Hiding a column
   takes its scroll position with it, so the reader's place is held here and
   handed back to it the moment the column returns. */
function setView(view) {
  rememberScroll();
  state.view = view;
  renderView();
  restoreScroll();
}

/* Only the columns on screen are read here: a hidden column reports a scroll
   position of its own — zero — and a column that is away right now keeps the
   place it was left at until it comes back. */
function rememberScroll() {
  if (state.view !== "target") {
    state.scroll.source = byId("source-text").scrollTop;
  }
  if (state.view !== "source") {
    state.scroll.target = byId("target-text").scrollTop;
  }
}

function restoreScroll() {
  byId("source-text").scrollTop = state.scroll.source;
  byId("target-text").scrollTop = state.scroll.target;
}

/* The two columns travel together while the switch above says so: a reader
   comparing a passage with its translation keeps their place without scrolling
   twice. The two are of different lengths — one language says the same thing in
   other words — so where they meet is the same fraction of each, not the same
   line. Off, each column is its own. */
function mirrorScroll(fromId, toId) {
  if (!state.sync || state.view !== "split") {
    return;
  }
  const from = byId(fromId);
  const to = byId(toId);
  const fromRoom = from.scrollHeight - from.clientHeight;
  const toRoom = to.scrollHeight - to.clientHeight;
  if (fromRoom <= 0 || toRoom <= 0) {
    return;
  }
  to.scrollTop = (from.scrollTop / fromRoom) * toRoom;
}

/* Dragging the divider sizes the pair, and the arrow keys size it a step at a
   time. The share is written as the number the stylesheet multiplies, so
   nothing here has to know about pixels or windows. */
function setSplit(percent) {
  state.split = Math.min(72, Math.max(28, percent));
  byId("panes").style.setProperty("--pane-split", state.split.toFixed(2));
  byId("pane-split").setAttribute("aria-valuenow", String(Math.round(state.split)));
}

function wireSplitter() {
  const split = byId("pane-split");
  const panes = byId("panes");
  const fromPointer = (event) => {
    const box = panes.getBoundingClientRect();
    if (box.width > 0) {
      setSplit(((event.clientX - box.left) / box.width) * 100);
    }
  };
  /* A pointer that is already gone — a drag that ended off the window, or an
     event with no real pointer behind it — cannot be captured or released.
     The drag is over at that point and nothing about the pair depends on the
     capture, so a refusal here is the whole of the answer. */
  const grab = (event, take) => {
    try {
      if (take) {
        split.setPointerCapture(event.pointerId);
      } else if (split.hasPointerCapture(event.pointerId)) {
        split.releasePointerCapture(event.pointerId);
      }
    } catch (error) {
      return;
    }
  };
  split.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    fromPointer(event);
    grab(event, true);
  });
  /* Captured, the pointer keeps to the divider however far the drag wanders, so
     the pair does not lose the drag halfway across the window. */
  split.addEventListener("pointermove", (event) => {
    if (split.hasPointerCapture(event.pointerId)) {
      fromPointer(event);
    }
  });
  split.addEventListener("pointerup", (event) => grab(event, false));
  split.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -2, ArrowRight: 2 }[event.key];
    if (!step) {
      return;
    }
    event.preventDefault();
    setSplit(state.split + step);
  });
}

/* ------------------------------------------------------------------ library */

/* The library is a column on a screen with room for one and a drawer over the
   document on a screen without: the same runs, two ways of being out of the
   way. Only the flag the stylesheet reads changes here, plus the words this
   control is announced by, plus the scrim — which exists only while the drawer
   is over the document. */
function renderRail() {
  const layout = byId("layout");
  if (!layout) {
    return; // the window changed shape before the page existed
  }
  layout.dataset.rail = state.drawer
    ? state.railOpen
      ? "open"
      : "closed"
    : state.rail;
  const shown = state.drawer ? state.railOpen : state.rail === "expanded";
  const toggle = byId("rail-toggle");
  toggle.setAttribute("aria-expanded", shown ? "true" : "false");
  toggle.setAttribute(
    "aria-label",
    shown ? "Thu gọn danh sách tài liệu" : "Mở danh sách tài liệu",
  );
  const scrim = byId("rail-scrim");
  show(scrim, state.drawer && state.railOpen);
  /* Reachable only while it is over the page and there is something behind it
     to go back to; as a column it hides nothing. */
  scrim.tabIndex = state.drawer && state.railOpen ? 0 : -1;
}

function closeRail() {
  if (!state.drawer || !state.railOpen) {
    return;
  }
  state.railOpen = false;
  renderRail();
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
    state.resumeSeen = 0;
    state.panesFor = runId;
    state.source = "idle";
    state.target = "idle";
    /* A new document is read from its own beginning, so the columns are put
       back to the top rather than left where the last one was left. */
    state.scroll.source = 0;
    state.scroll.target = 0;
    /* Another document's chunks are not this one's. */
    state.partialText = "";
    state.partialPieces = 0;
    state.partialDone = -1;
    state.sourceText = "";
    state.targetText = "";
    state.sourceNote = "";
    state.targetNote = "";
    state.follow = true;
    state.chunkFilter = "all";
    state.chunkOpen = null;
    resetLog();
  }
  renderJobs();
  try {
    await loadDetail(runId);
  } catch (error) {
    renderBanner("error", `Không mở được lượt dịch: ${error.message}`);
    return;
  }
  /* Drawing a document's text is what resets a column's scroll, so the place
     the reader is owed — the top, for a document just opened — is set here. */
  restoreScroll();
  /* A run that is already deep into a document shows what it has translated
     without waiting for the next poll. */
  refreshPartial();
  const run = state.detail.run;
  state.seenStatus = run.status;
  if (!isFinal(run.status)) {
    pollRun(runId);
  } else if (!state.events.length) {
    /* A run that died before its first event has nothing to replay; its own
       output is the only explanation, and the server reports it as a problem
       the log picks up. */
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
    refreshPartial();
  }

  const changed = payload.status !== state.seenStatus;
  state.seenStatus = payload.status;
  /* A resumed run whose checkpoint store had nothing matching reports it as its
     own fact — not a fault of this run, not a failure of the previous one. A
     light notice, only for the running case, so it leaves the moment the run
     does its work or finishes. Named `resume` rather than `resumeInfo` so it
     does not shadow the resumeInfo() helper called in this same function.

     It is raised once for the attempt it describes. The record stays in the
     log for the whole run, so re-stamping it on every poll made the banner
     blink in and out for as long as the run went on — a warning about the
     start of the attempt, chasing a reader through the hour that followed. */
  const resume = resumeInfo(state.events);
  if (
    resume &&
    resume.ts &&
    resume.ts !== state.resumeSeen &&
    payload.status === "running" &&
    resume.requested &&
    resume.resumed === 0 &&
    !bannerActive()
  ) {
    state.resumeSeen = resume.ts;
    renderBanner("warn", resumeNotice(resume));
  }
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
   "Hủy" and Escape resolve false — the same as cancelling.

   The verdict comes from the dialog's own `submit` and `cancel`, not from its
   `close`: the desktop webview this page is opened in closes the dialog on a
   `method="dialog"` submit without ever dispatching `close`, and a delete that
   waits for an event which never arrives leaves the reader clicking the trash
   at a run that stays on screen until they reload the page by hand. The submit
   is answered here — the browser's own close is prevented — so the dialog shuts
   on the very event that decides the answer. `close` is still listened to, so a
   host that does fire it is answered too; whichever arrives first wins. */
function confirmDelete() {
  const dlg = byId("confirm-delete");
  if (typeof dlg.showModal !== "function") {
    return Promise.resolve(false); // no modal support: never delete on a guess
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (answer) => {
      if (settled) {
        return;
      }
      settled = true;
      dlg.removeEventListener("submit", onSubmit);
      dlg.removeEventListener("cancel", onCancel);
      dlg.removeEventListener("close", onClosed);
      if (dlg.open) {
        dlg.close(answer);
      }
      resolve(answer === "delete");
    };
    /* Enter inside the dialog submits with the first button, "Hủy" — a default
       that cancels, which is the safe way round for an action that cannot be
       undone. */
    const onSubmit = (event) => {
      event.preventDefault();
      const button = event.submitter;
      finish(button && button.value === "delete" ? "delete" : "cancel");
    };
    const onCancel = () => finish("cancel"); // Escape
    const onClosed = () => finish(dlg.returnValue);
    dlg.addEventListener("submit", onSubmit);
    dlg.addEventListener("cancel", onCancel);
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
  /* The erase reaches further than the one row: the list, the filters, the open
     panel and the timers watching that run all held something about it, and the
     row was only the part of it that was on screen. All of it is dropped now,
     not at the next visit: the server has answered, so the row that was just
     deleted goes at once — leaving it there and asking the page to reload itself
     is what had the reader pressing F5 to see it gone. */
  clearTimeout(state.timer);
  clearTimeout(state.listTimer);
  if (state.open === runId) {
    state.open = null;
    state.detail = null;
    state.events = [];
    state.next = 0;
    state.seenStatus = null;
    state.hadEvents = false;
    state.resumeSeen = 0;
    state.panesFor = null;
    state.source = "idle";
    state.target = "idle";
    /* A new document is read from its own beginning, so the columns are put
       back to the top rather than left where the last one was left. */
    state.scroll.source = 0;
    state.scroll.target = 0;
    /* Another document's chunks are not this one's. */
    state.partialText = "";
    state.partialPieces = 0;
    state.partialDone = -1;
    state.sourceText = "";
    state.targetText = "";
    state.sourceNote = "";
    state.targetNote = "";
    renderEmptyJob();
  }
  state.runs = state.runs.filter((run) => run.run_id !== runId);
  renderFilters();
  renderJobs();
  hideBanner();
  /* Then read the list back from the server, so what is left on screen is the
     server's own answer and not a patch that outlived the run. */
  await loadRuns();
}

/* "Tiếp tục" continues the run the reader is already watching. The page sends
   nothing but that run's id: its document, languages, output format and
   Translator model are all stored on the run itself, so there is no second set
   of settings to drift out of step with the one that produced the
   checkpoints. */
async function resumeRun(runId) {
  try {
    await post("/api/runs", { resume: true, run_id: runId });
  } catch (error) {
    renderBanner("error", `Không tiếp tục được lượt dịch: ${error.message}`);
    return;
  }
  /* The run keeps its id, so the view does not move: it picks the continued
     attempt up from the same log, which by now holds both attempts. */
  await loadRuns();
  await openJob(runId);
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

async function uploadFile(file) {
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
  /* A re-upload of the same document replaces it: the server overwrites, so a
     repeat upload just refreshes the bytes rather than asking for consent. */
  const route = `/api/inputs/${encodeURIComponent(name)}`;
  try {
    await sendFile(route, file, (ratio) => {
      byId("upload-fill").style.width = `${Math.round(ratio * 100)}%`;
    });
    state.uploaded = name;
    byId("upload-bar").dataset.state = "ok";
    byId("upload-fill").style.width = "100%";
    dropMessage("idle", `Đã tải ${name} lên — bấm “Bắt đầu dịch”.`);
    byId("new-job-start").disabled = false;
  } catch (error) {
    byId("upload-bar").dataset.state = "fail";
    dropMessage("bad", `Không tải lên được: ${error.message}`);
  }
}

function handleDropped(file) {
  if (!file) {
    return;
  }
  uploadFile(file);
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
  formAlert("Đang nhập tài liệu ...", "ok");
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
  /* Clicking a notice waves it away instead of waiting out its timer. */
  byId("banner").addEventListener("click", hideBanner);

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

  byId("view-split").addEventListener("click", () => setView("split"));
  byId("view-source").addEventListener("click", () => setView("source"));
  byId("view-target").addEventListener("click", () => setView("target"));

  /* The two tools beside the layout switch: whether the columns travel
     together, and whether the pair takes the whole window. */
  byId("view-sync").addEventListener("click", () => {
    state.sync = !state.sync;
    renderView();
  });
  byId("view-zoom").addEventListener("click", () => {
    state.zoom = !state.zoom;
    renderView();
  });
  wireSplitter();
  byId("source-text").addEventListener("scroll", () =>
    mirrorScroll("source-text", "target-text"),
  );
  byId("target-text").addEventListener("scroll", () =>
    mirrorScroll("target-text", "source-text"),
  );

  /* The library: folded to a column of tiles where there is room for a column,
     opened and shut as a drawer where there is not. */
  byId("rail-toggle").addEventListener("click", () => {
    if (state.drawer) {
      state.railOpen = !state.railOpen;
    } else {
      state.rail = state.rail === "expanded" ? "collapsed" : "expanded";
    }
    renderRail();
  });
  byId("rail-scrim").addEventListener("click", closeRail);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeRail();
    }
  });

  byId("tab-result").addEventListener("click", () => {
    state.tab = "result";
    renderTabs();
  });
  byId("tab-status").addEventListener("click", () => {
    state.tab = "status";
    /* Opening the list asks to see the run, so it starts out following it. */
    state.follow = true;
    renderTabs();
    /* Switching to the grid draws it at once, not at the next poll: the reader
       asked to see it now. */
    renderChunks(state.detail);
  });
  byId("tab-history").addEventListener("click", () => {
    state.tab = "history";
    renderTabs();
    /* The log is append-only, so this draws whatever has happened since it was
       last open and leaves the lines already drawn alone. */
    renderLog();
  });
  byId("tab-report").addEventListener("click", () => {
    state.tab = "report";
    renderTabs();
    renderReport(state.detail);
  });
  /* A scroll the reader made is the last word on where the list sits. Following
     re-starts by itself once the running chunk is back in view, so no control
     has to exist for it. */
  byId("chunks").addEventListener("scroll", () => {
    state.follow = followRunning();
  });
  byId("chunk-filter-all").addEventListener("click", () => {
    state.chunkFilter = "all";
    renderChunks(state.detail);
  });
  byId("chunk-filter-fail").addEventListener("click", () => {
    state.chunkFilter = "fail";
    renderChunks(state.detail);
  });
}

/* ------------------------------------------------------------- first paint */

/* The document arrives under the splash, and hands it over here: what a reader
   is waiting for is the first list of runs, not the files. So the mark stays up
   until that list has landed — and at least long enough to be seen, because a
   mark that appears and leaves inside one frame reads as a flicker, not as a
   start. Neither wait is a progress report: the fetch reports its own failures
   in the banner, and the splash leaves either way. */
const SPLASH_FADE_MS = 400;

/* The stage runs on its own clock: the mark settles in first, then the scan
   sweep runs, then the fade begins. The splash stays up for a full cycle even
   when the first list lands quickly, so the effect is never skipped. */
const SPLASH_LOGO_MS = 300; // the mark settles in
const SPLASH_SCAN_MS = 1500; // the sweep runs after the mark is up
const SPLASH_HOLD_MS = 400; // the mark holds before the fade begins

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* The stage's own timeline: the sweep runs, then the mark settles. A splash
   that is already leaving is not redrawn. */
function startSplash() {
  const stage = byId("preloader");
  if (!stage || stage.classList.contains("preloader--leaving")) {
    return;
  }
  const overlay = byId("preloader-scan-overlay");
  const line = byId("preloader-scan-line");
  const logo = byId("preloader-logo");
  /* The mark settles in first. */
  setTimeout(() => {
    if (stage.classList.contains("preloader--leaving")) {
      return;
    }
    if (logo) {
      logo.classList.add("show");
    }
  }, SPLASH_LOGO_MS);
  /* Then the sweep runs. */
  setTimeout(() => {
    if (stage.classList.contains("preloader--leaving")) {
      return;
    }
    if (overlay) {
      overlay.classList.add("active");
    }
    if (line) {
      line.classList.add("active");
    }
    /* The mark leaves with the light rather than with the overlay's class: the
       sweep clears itself away, and a fade tied to that would run backwards. */
    if (logo) {
      logo.classList.add("scanning");
    }
    /* After the sweep, hold for a beat, then fade. */
    setTimeout(() => {
      if (stage.classList.contains("preloader--leaving")) {
        return;
      }
      if (overlay) {
        overlay.classList.remove("active");
      }
      wait(SPLASH_HOLD_MS).then(() => {
        if (!stage.classList.contains("preloader--leaving")) {
          hideSplash();
        }
      });
    }, 1600); // the sweep's own duration
  }, SPLASH_SCAN_MS);
}

function hideSplash() {
  const stage = byId("preloader");
  if (!stage || stage.hidden) {
    return;
  }
  stage.classList.add("preloader--leaving");
  state.splashFade = setTimeout(() => {
    stage.classList.remove("preloader--leaving");
    show(stage, false);
    state.splashFade = null;
  }, SPLASH_FADE_MS);
}

async function boot() {
  startSplash();
  wire();
  renderRail();
  renderEmptyJob();
  renderFilters();
  renderView();
  renderTabs();
  /* The splash leaves on its own clock (see startSplash): the mark has to be
     seen. Fetching the first list runs in the background; when it is ready
     while the splash still holds, it is re-rendered the moment the splash goes. */
  loadRuns();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
