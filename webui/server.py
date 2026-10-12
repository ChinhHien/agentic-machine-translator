"""HTTP surface for the web UI: a pure router plus a thin socket handler.

``Api.handle`` takes plain data and returns plain data, so nearly every test
can run without opening a port; :class:`Handler` only reads a request, calls
the router and writes the response back. The server binds ``127.0.0.1`` on
purpose: it can start and kill translation runs, so it must not be reachable
from the network.

What this surface deliberately does **not** have is any route that reads or
writes the system configuration. There is no ``/api/config``: the application
has no view of ``.env`` — not a value, not a key name, not whether a credential
is set. Runs inherit the server's environment, and a human changes that from a
terminal. The only translation preferences the API accepts are the ones that
belong to a single run (languages, domain, glossary, chunk budget, output
format, resume).

Everything a response quotes back — run notes, step errors, the child's own
output — is passed through :func:`_redact` first, so a provider error that
echoes a credential cannot travel to a browser.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from webui.paths import REPO_ROOT, is_within, safe_child
from webui.runs import LOGS_DIR, RunManager, RunPlanError
from webui.store import GlossaryStore, RunRecord, RunsStore, SkillsStore

logger = logging.getLogger("webui")

#: Only loopback, and only this port by default.
HOST = "127.0.0.1"
DEFAULT_PORT = 8787

#: Static assets are served by extension, never by "the file exists".
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}

JSON_TYPE = "application/json; charset=utf-8"

#: Methods that can change state on disk, and so need a same-origin check.
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Query values that mean "yes" for a boolean option such as `?overwrite=1`.
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

#: Directory holding the documents a run may translate.
INPUT_DIR = "input"

#: The only input the pipeline accepts: a PDF with a text layer. A scanned PDF
#: is refused by the ingest layer (no OCR), and Markdown/text input no longer
#: exists at all, so neither belongs in the launcher or in an upload.
INPUT_SUFFIX = ".pdf"

#: Refuse an upload larger than this before it is read into memory. Generous
#: for a scanned book, small enough that a mistake cannot exhaust the machine.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

#: Every PDF starts with this, within the first kilobyte (some writers put a
#: byte-order mark or a comment in front of it).
PDF_MAGIC = b"%PDF-"

#: Markdown, for the route that returns the translated document.
MARKDOWN_TYPE = "text/markdown; charset=utf-8"

#: The rebuilt, layout-preserving PDF.
PDF_TYPE = "application/pdf"

#: Several artefacts at once.
ZIP_TYPE = "application/zip"

#: What a run may ask to be written (`--format`).
OUTPUT_FORMATS = frozenset({"md", "pdf", "both"})

#: Fields the run list shows; the command line and pid stay server-side. A run's
#: command line is a developer detail — it names the interpreter, the absolute
#: paths and the flags — and the page has no use for it.
RUN_SUMMARY_FIELDS = (
    "run_id",
    "started",
    "status",
    "file",
    "src",
    "tgt",
    "domain",
    "chunks",
    "exit_code",
    "output_path",
    "note",
)

#: Event statuses that mean something went wrong, for the "problems" block.
PROBLEM_STATUSES = frozenset({"fail", "retry"})

#: A credential-shaped run of characters. OpenAI's own 401 message quotes the
#: first characters of the key it rejected, so error text is not trustworthy.
_SECRET_RE = re.compile(r"\b(?:sk|pk|rk|api|key|token)[-_][A-Za-z0-9_\-]{6,}\b", re.IGNORECASE)

#: How much of a child's output to quote when a run died before any event.
TAIL_CHARS = 4000

#: How long a memory report is reused, in seconds. Reading the reasoning bank
#: costs a ChromaDB import; a drawer being reopened should not pay it twice.
MEMORY_TTL = 60.0


def _redact(text: str) -> str:
    """Text safe to hand a browser: credential-shaped runs are masked."""
    if not text:
        return ""
    return _SECRET_RE.sub("***", text)


def _content_disposition(name: str) -> str:
    """An attachment header for a name that may not be ASCII.

    Document names are frequently Vietnamese, and a bare ``filename="bản.pdf"``
    is not valid in a header; the percent-encoded ``filename*`` form is what
    every current browser reads.
    """
    clean = name.replace('"', "").replace("\\", "").replace("/", "")
    encoded = quote(clean, safe="")
    return f"attachment; filename=\"{encoded}\"; filename*=UTF-8''{encoded}"


@dataclass
class Response:
    """One HTTP response, in the plainest possible shape."""

    status: int
    body: bytes
    content_type: str
    #: Set only for a download; the handler turns it into a header.
    disposition: str | None = None


def json_response(payload: object, status: int = 200) -> Response:
    """A JSON response; non-ASCII text stays readable instead of escaped."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return Response(status, body, JSON_TYPE)


def error(status: int, message: str) -> Response:
    """A JSON error, so the front end never has to parse HTML for a failure."""
    return json_response({"error": _redact(message)}, status)


def _flagged(query: dict[str, list[str]], name: str) -> bool:
    """Whether a query option is present and means yes."""
    return any(value.strip().lower() in TRUE_VALUES for value in query.get(name, []))


def _first(query: dict[str, list[str]], name: str) -> str | None:
    """The first value of a query option, or ``None`` when it was absent."""
    values = query.get(name) or []
    return values[0] if values else None


def _json_object(body: bytes) -> dict[str, Any] | None:
    """A JSON object from a request body; ``None`` for anything else.

    A malformed body is a client error the caller reports as 400 — it must not
    become a 500, and it must not be silently treated as empty either.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _public_record(record: RunRecord) -> dict[str, Any]:
    """A record's fields as they may travel: free text is redacted first.

    A record's ``note`` is written from whatever the run reported, so it can
    carry a provider's error message verbatim; the command line is a developer
    detail that never needed to leave the server either.
    """
    data = record.to_dict()
    data["note"] = _redact(str(data.get("note") or ""))
    data["command"] = [_redact(str(part)) for part in data.get("command") or []]
    return data


class Api:
    """Routes requests to the stores and the run manager."""

    def __init__(
        self,
        root: Path,
        manager: RunManager,
        runs: RunsStore,
        skills: SkillsStore,
        glossaries: GlossaryStore,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.root = Path(root)
        self.manager = manager
        self.runs = runs
        self.skills = skills
        self.glossaries = glossaries
        self.port = port
        #: `(path, size, mtime) -> extracted text`, so flipping between two
        #: panes or reloading the page does not re-read a whole book.
        self._source_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        #: `(read_at, payload)` for the memory report (see :data:`MEMORY_TTL`).
        self._memory_cache: tuple[float, dict[str, Any]] | None = None

    # Routing -----------------------------------------------------------
    def handle(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: bytes,
        headers: dict[str, str],
    ) -> Response:
        """Dispatch one request. Unknown routes are a JSON 404, never HTML."""
        method = method.upper()
        # Before routing: a page on another origin must not be able to start,
        # stop or overwrite anything, however it addresses it.
        if method in MUTATING_METHODS and not self._origin_allowed(headers):
            return error(403, "refused: request must come from the local UI")
        if path == "/" and method == "GET":
            return self._index()
        if path.startswith("/static/") and method == "GET":
            return self._static(path[len("/static/") :])
        if path == "/api/health" and method == "GET":
            return json_response({"ok": True})
        return self._api(method, path, query, body)

    def _api(self, method: str, path: str, query: dict[str, list[str]], body: bytes) -> Response:
        """The API routes, in one table, with no configuration among them."""
        if method == "GET" and path.rstrip("/") == "/api/memory":
            return self._memory()
        if path.rstrip("/") == "/api/runs":
            if method == "GET":
                return self._runs()
            if method == "POST":
                return self._start_run(body)
        if path.startswith("/api/runs/"):
            answer = self._run_route(method, path[len("/api/runs/") :], query)
            if answer is not None:
                return answer
        if path.rstrip("/") == "/api/inputs" and method == "GET":
            return self._inputs()
        if path.startswith("/api/inputs/") and method == "PUT":
            # Unquote *before* validating: otherwise `..%2F.env` would look
            # like a harmless single-segment name and slip past `safe_child`.
            return self._upload(unquote(path[len("/api/inputs/") :]), query, body)
        if path.startswith("/api/glossaries"):
            answer = self._glossary_route(method, path, body)
            if answer is not None:
                return answer
        if path.startswith("/api/skills") and method == "GET":
            answer = self._skill_route(path)
            if answer is not None:
                return answer
        return error(404, f"no such route: {method} {path}")

    # Origin --------------------------------------------------------------
    @property
    def allowed_origins(self) -> frozenset[str]:
        """The only pages allowed to mutate anything: the UI served here."""
        return frozenset(
            {
                f"http://127.0.0.1:{self.port}",
                f"http://localhost:{self.port}",
            }
        )

    @staticmethod
    def _header(headers: dict[str, str], name: str) -> str | None:
        """A header by name, whatever case the client happened to use."""
        wanted = name.lower()
        for key, value in headers.items():
            if key.lower() == wanted:
                return value
        return None

    def _origin_allowed(self, headers: dict[str, str]) -> bool:
        origin = self._header(headers, "origin")
        return origin is not None and origin in self.allowed_origins

    # Inputs ---------------------------------------------------------------
    def _inputs(self) -> Response:
        """Documents available to translate, newest metadata included.

        Only PDFs are listed. A file the pipeline would refuse is worse than no
        entry at all: it looks selectable and then fails at the first step.
        """
        directory = self.root / INPUT_DIR
        files = []
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() == INPUT_SUFFIX:
                    info = path.stat()
                    files.append({"name": path.name, "size": info.st_size, "mtime": info.st_mtime})
        return json_response({"files": files})

    def _upload(self, name: str, query: dict[str, list[str]], body: bytes) -> Response:
        """Write a PDF into ``input/``, refusing to clobber one by accident.

        Validation runs in the order that produces the clearest message: a name
        that is really a path is refused as a path attempt, then what the bytes
        are, then whether it would overwrite something. Nothing is written until
        all of that passes.
        """
        try:
            path = safe_child(self.root / INPUT_DIR, name)
        except ValueError:
            return error(403, f"refused: {name!r} is not a plain file name")
        if path.suffix.lower() != INPUT_SUFFIX:
            return error(403, f"refused: only {INPUT_SUFFIX} input is supported")
        if not body:
            return error(400, "empty upload")
        if len(body) > MAX_UPLOAD_BYTES:
            return error(
                413,
                f"file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
        if PDF_MAGIC not in body[:1024]:
            return error(422, "this is not a PDF file (no %PDF- header)")
        if path.exists() and not _flagged(query, "overwrite"):
            return error(409, f"{name} already exists; pass ?overwrite=1 to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        logger.info("uploaded %s (%d bytes)", name, len(body))
        return json_response({"name": name, "size": len(body)})

    # Runs ----------------------------------------------------------------
    def _runs(self) -> Response:
        """Every run, newest first, with the status reconciled to reality."""
        summaries = []
        for record in self.runs.list_records():
            if record.status == "running":
                try:
                    record = self.manager.status(record.run_id)
                except KeyError:  # vanished between listing and checking
                    continue
            summaries.append(self._summary(record))
        return json_response({"runs": summaries})

    def _summary(self, record: RunRecord) -> dict[str, Any]:
        """One list row: the record's public fields plus what is downloadable."""
        data = _public_record(record)
        summary = {field: data[field] for field in RUN_SUMMARY_FIELDS}
        summary["has_md"] = self._md_path(record) is not None
        summary["has_pdf"] = self._pdf_path(record) is not None
        return summary

    def _run_route(self, method: str, rest: str, query: dict[str, list[str]]) -> Response | None:
        """``/api/runs/<id>`` and its sub-resources; ``None`` when unknown."""
        parts = [unquote(part) for part in rest.split("/")]
        run_id = parts[0]
        if len(parts) == 1 and method == "GET":
            return self._run_detail(run_id)
        if len(parts) == 1 and method == "DELETE":
            return self._delete_run(run_id)
        if len(parts) == 2 and method == "GET":
            if parts[1] == "events":
                return self._run_events(run_id, _first(query, "after"))
            if parts[1] == "output":
                return self._run_output(run_id, _flagged(query, "download"))
            if parts[1] == "pdf":
                return self._run_pdf(run_id, _flagged(query, "download"))
            if parts[1] == "bundle":
                return self._run_bundle(run_id)
            if parts[1] == "source":
                return self._run_source(run_id)
        if len(parts) == 2 and method == "POST" and parts[1] == "stop":
            return self._stop_run(run_id)
        return None

    def _run_detail(self, run_id: str) -> Response:
        """Everything the job view needs about one run, in one round trip."""
        try:
            record = self.manager.status(run_id)
        except KeyError:
            return error(404, f"no such run: {run_id}")
        md, pdf = self._md_path(record), self._pdf_path(record)
        return json_response(
            {
                "run": _public_record(record),
                "md": md is not None,
                "md_path": str(md) if md else "",
                "pdf": pdf is not None,
                # The path itself, not only the flag: a PDF-only run never writes
                # the Markdown the record points at, so the page would otherwise
                # name a file that is not there.
                "pdf_path": str(pdf) if pdf else "",
                "artifacts": self._artifacts(record),
                "problems": self._problems(record),
                "routes": self._routes(record),
            }
        )

    def _routes(self, record: RunRecord) -> dict[str, int]:
        """How many chunks took each route, from the run's own decisions.

        Same reading as :meth:`_problems`: the decisions the pipeline recorded,
        reduced to a tally. A run with no decisions at all reports zeros rather
        than pretending every chunk was standard.
        """
        counts = {"fast": 0, "standard": 0, "deep": 0}
        try:
            events, _next, _malformed = self.manager.events(record.run_id, 0)
        except Exception:  # noqa: BLE001 - a broken log must not hide the run
            return counts
        fast: set[str] = set()
        refined: set[str] = set()
        seen: set[str] = set()
        for event in events:
            if str(event.get("type") or "") != "decision":
                continue
            chunk_id = str(event.get("chunk_id") or "")
            seen.add(chunk_id)
            kind = str(event.get("kind") or "")
            if kind == "fast_path" and event.get("chosen"):
                fast.add(chunk_id)
            elif kind == "refine_gate" and event.get("chosen"):
                refined.add(chunk_id)
        for chunk_id in seen:
            if chunk_id in fast:
                counts["fast"] += 1
            elif chunk_id in refined:
                counts["deep"] += 1
            else:
                counts["standard"] += 1
        return counts

    def _md_path(self, record: RunRecord) -> Path | None:
        """The run's Markdown, when it exists inside the repository."""
        if not record.output_path:
            return None
        path = Path(record.output_path)
        if not is_within(self.root, path) or not path.is_file():
            return None
        return path

    def _pdf_path(self, record: RunRecord) -> Path | None:
        """The rebuilt PDF on disk, if the run wrote one.

        Derived from the run's Markdown path — ``output_paths`` gives both from
        one base — and still checked against the repository, so a hand-edited
        record cannot point the reader at an arbitrary file.
        """
        if not record.output_path:
            return None
        path = Path(record.output_path).with_suffix(".pdf")
        if not is_within(self.root, path) or not path.is_file():
            return None
        return path

    def _artifacts(self, record: RunRecord) -> list[dict[str, Any]]:
        """The files this run wrote, newest knowledge first, for the UI."""
        found = []
        for kind, path in (("md", self._md_path(record)), ("pdf", self._pdf_path(record))):
            if path is None:
                continue
            found.append(
                {
                    "kind": kind,
                    "name": path.name,
                    "size": path.stat().st_size,
                    "path": str(path),
                }
            )
        return found

    def _problems(self, record: RunRecord) -> list[dict[str, Any]]:
        """What went wrong, in the run's own words.

        Two sources, because they explain different failures: the event log
        knows which chunk and which step, and the child's own output is the
        only place a crash *before* the first event is explained (a missing
        credential at import time, for instance).
        """
        problems: list[dict[str, Any]] = []
        try:
            events, _next, _malformed = self.manager.events(record.run_id, 0)
        except Exception:  # noqa: BLE001 - a broken log must not hide the run
            events = []
        for event in events:
            status = str(event.get("status") or "")
            if status not in PROBLEM_STATUSES:
                continue
            problems.append(
                {
                    "step": str(event.get("step") or ""),
                    "chunk_id": str(event.get("chunk_id") or ""),
                    "status": status,
                    # The pipeline writes a cause from its own closed set; this
                    # server copies the word, it does not interpret it (that is
                    # why it can stay free of any polyglot_lc import).
                    "cause": str(event.get("cause") or ""),
                    "message": _redact(str(event.get("error") or event.get("message") or ""))[
                        :TAIL_CHARS
                    ],
                }
            )
        if record.status == "failed" and not problems:
            tail = _redact(self.runs.tail_output(record.run_id))[:TAIL_CHARS]
            if tail:
                problems.append({"step": "run", "chunk_id": "", "status": "fail", "message": tail})
        return problems

    def _run_events(self, run_id: str, after: str | None) -> Response:
        """Events from index *after* on, plus the run's current state.

        The state is resolved first, so the last poll of a finished run still
        reports how it ended rather than its final intermediate status.
        """
        try:
            record = self.manager.status(run_id)
        except KeyError:
            return error(404, f"no such run: {run_id}")
        try:
            index = max(0, int(after or 0))
        except (TypeError, ValueError):
            return error(400, "after must be a whole number")
        events, next_index, malformed = self.manager.events(run_id, index)
        return json_response(
            {
                "events": [_redacted_event(event) for event in events],
                "next": next_index,
                "status": record.status,
                "exit_code": record.exit_code,
                "malformed": malformed,
                # Without this the UI cannot tell "nothing has happened yet" from
                # "this run will never report anything", and would sit silent.
                "has_events": self.manager.has_events(run_id),
            }
        )

    def _start_run(self, body: bytes) -> Response:
        data = _json_object(body)
        if data is None:
            return error(400, "body must be a JSON object")
        file = str(data.get("file") or "").strip()
        if not file:
            return error(400, "file is required")
        raw_tokens = data.get("max_tokens")
        try:
            max_tokens = int(raw_tokens) if raw_tokens not in (None, "") else None
        except (TypeError, ValueError):
            return error(400, "max_tokens must be a whole number")
        output_format = str(data.get("format") or "md").strip().lower()
        if output_format not in OUTPUT_FORMATS:
            return error(400, "format must be md, pdf or both")
        glossary = str(data.get("glossary") or "").strip() or None
        if glossary is not None:
            try:
                path = self.glossaries.path_for(glossary)
            except ValueError as exc:
                return error(403, f"refused: {exc}")
            if not path.is_file():
                return error(422, f"no such glossary: {glossary}")
            glossary = str(path)
        try:
            record = self.manager.start(
                file=file,
                src=str(data.get("src") or "auto"),
                tgt=str(data.get("tgt") or "vi"),
                domain=str(data.get("domain") or "general"),
                glossary=glossary,
                max_tokens=max_tokens,
                # Kept for the test suite and for a demo deployment driven from
                # the terminal; the page never sends it (see the module doc).
                mock=bool(data.get("mock")),
                output_format=output_format,
                resume=bool(data.get("resume")),
            )
        except RunPlanError as exc:
            # The document could not be read or planned: nothing was started.
            return error(422, str(exc))
        except ValueError as exc:
            return error(403, f"refused: {exc}")
        return json_response(
            {
                "run_id": record.run_id,
                "chunks": record.chunks,
                "output_path": record.output_path,
                "artifacts": self._artifacts(record),
            },
            201,
        )

    def _stop_run(self, run_id: str) -> Response:
        """Stop a run; the answer carries the state it actually reached."""
        try:
            record = self.manager.stop(run_id)
        except KeyError:
            return error(404, f"no such run: {run_id}")
        except RuntimeError as exc:
            return error(409, f"cannot stop: {exc}")
        return json_response({"status": record.status, "exit_code": record.exit_code})

    def _delete_run(self, run_id: str) -> Response:
        """Erase a run's history and its output from disk.

        Deleting is not reversible: the log, the events and both translated
        files are gone for good, so the front end asks before it sends this.
        The run's *input* PDF is not touched (it may belong to another run).
        """
        try:
            self.manager.delete(run_id)
        except KeyError:
            return error(404, f"no such run: {run_id}")
        return json_response({"deleted": run_id})

    # Artefacts -----------------------------------------------------------
    def _run_output(self, run_id: str, download: bool = False) -> Response:
        """The translated document, once it exists.

        The path comes from our own record, but it is still checked against the
        repository: a hand-edited `run.json` must not turn this into a reader
        for arbitrary files.
        """
        record = self.runs.read_record(run_id)
        if record is None:
            return error(404, f"no such run: {run_id}")
        if not record.output_path:
            return error(404, f"run {run_id} has no output path")
        if not is_within(self.root, Path(record.output_path)):
            return error(403, "refused: output path is outside the repository")
        path = self._md_path(record)
        if path is None:
            return error(404, "this run has not written its output yet")
        return Response(
            200,
            path.read_bytes(),
            MARKDOWN_TYPE,
            _content_disposition(path.name) if download else None,
        )

    def _run_pdf(self, run_id: str, download: bool = False) -> Response:
        """The rebuilt PDF, when the run asked for one."""
        record = self.runs.read_record(run_id)
        if record is None:
            return error(404, f"no such run: {run_id}")
        if not record.output_path:
            return error(404, f"run {run_id} has no output path")
        if not is_within(self.root, Path(record.output_path)):
            return error(403, "refused: output path is outside the repository")
        path = self._pdf_path(record)
        if path is None:
            return error(404, "this run has no rebuilt PDF")
        return Response(
            200, path.read_bytes(), PDF_TYPE, _content_disposition(path.name) if download else None
        )

    def _run_bundle(self, run_id: str) -> Response:
        """Every artefact of one run in a single download.

        A run is one job with up to two results, and a reader who wants both
        should not have to fetch them one at a time. The archive is built in
        memory from exactly the files that exist — nothing is generated or
        renamed to make it look complete.
        """
        record = self.runs.read_record(run_id)
        if record is None:
            return error(404, f"no such run: {run_id}")
        files = [
            path for path in (self._md_path(record), self._pdf_path(record)) if path is not None
        ]
        if not files:
            return error(404, "this run has not written anything yet")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.name)
        return Response(
            200, buffer.getvalue(), ZIP_TYPE, _content_disposition(f"{record.run_id}-ket-qua.zip")
        )

    def _run_source(self, run_id: str) -> Response:
        """The source document's own text, for the pane a reader compares to.

        Extraction is the pipeline's own (through the probe), so the text shown
        is the text that was translated, and it is cached by file identity:
        re-reading a 40-page book on every pane switch would make the page feel
        broken.
        """
        record = self.runs.read_record(run_id)
        if record is None:
            return error(404, f"no such run: {run_id}")
        try:
            path = self.manager.input_path(record.file)
        except ValueError:
            return error(403, f"refused: {record.file!r} is not a plain file name")
        if not path.is_file():
            return error(404, f"the run's input file is gone: {record.file}")
        info = path.stat()
        key = (str(path), info.st_size, info.st_mtime_ns)
        cached = self._source_cache.get(key)
        if cached is not None:
            return json_response({**cached, "cached": True})
        try:
            payload = self.manager.probe(["--text", str(path)])
        except RunPlanError as exc:
            return error(422, f"could not read the source document: {exc}")
        if not isinstance(payload, dict) or "text" not in payload:
            return error(502, "the extractor returned no text")
        if len(self._source_cache) >= 4:  # bound it: a book is megabytes
            self._source_cache.pop(next(iter(self._source_cache)))
        self._source_cache[key] = payload
        return json_response({**payload, "cached": False})

    # Memory ---------------------------------------------------------------
    def _memory(self) -> Response:
        """What the translation memory holds — read-only, through the probe.

        The report is a sample of the reasoning bank (see ``probe --memory``):
        it says how many traces exist per language pair and what the recent
        strategies were. Nothing here can write to the bank.
        """
        now = time.monotonic()
        if self._memory_cache is not None and now - self._memory_cache[0] < MEMORY_TTL:
            return json_response({"memory": self._memory_cache[1], "cached": True})
        try:
            payload = self.manager.probe(["--memory"])
        except RunPlanError as exc:
            return error(502, f"could not read the translation memory: {exc}")
        if not isinstance(payload, dict):
            return error(502, "the memory report was not an object")
        self._memory_cache = (now, payload)
        return json_response({"memory": payload, "cached": False})

    # Glossaries and skills -------------------------------------------------
    def _glossary_route(self, method: str, path: str, body: bytes) -> Response | None:
        if path.rstrip("/") == "/api/glossaries" and method == "GET":
            return json_response({"files": self.glossaries.list_files()})
        if not path.startswith("/api/glossaries/"):
            return None
        name = unquote(path[len("/api/glossaries/") :])
        try:
            target = self.glossaries.path_for(name)
        except ValueError as exc:
            return error(403, f"refused: {exc}")
        if method == "GET":
            if not target.is_file():
                return error(404, f"no such glossary: {name}")
            return json_response({"name": target.name, "pairs": self.glossaries.read(name)})
        if method == "PUT":
            data = _json_object(body)
            if data is None:
                return error(400, "body must be a JSON object")
            pairs = data.get("pairs")
            if not isinstance(pairs, list) or not all(isinstance(row, list) for row in pairs):
                return error(400, "pairs must be a list of [source, target] rows")
            self.glossaries.write(name, pairs)
            return json_response({"name": target.name, "pairs": pairs})
        return None

    def _skill_route(self, path: str) -> Response | None:
        if path.rstrip("/") == "/api/skills":
            return json_response({"skills": self.skills.list_skills()})
        if not path.startswith("/api/skills/"):
            return None
        parts = [unquote(part) for part in path[len("/api/skills/") :].split("/")]
        if len(parts) != 2:
            return None
        pair, domain = parts
        try:
            skill = self.skills.get_skill(pair, domain)
        except ValueError:  # a name that could not be a real pair directory
            skill = None
        if skill is None:
            return error(404, f"no skill for {pair}/{domain}")
        return json_response({"skill": skill, "history": self.skills.history(pair, domain)})

    # Static ------------------------------------------------------------
    @property
    def static_dir(self) -> Path:
        """The asset directory, laid out as ``webui/static`` under the root."""
        return self.root / "webui" / "static"

    def _index(self) -> Response:
        return self._read_static("index.html")

    def _static(self, name: str) -> Response:
        return self._read_static(name)

    def _read_static(self, name: str) -> Response:
        """Serve one whitelisted asset, or a JSON 404.

        A request for an asset type we do not serve (a ``.py`` next to the
        assets) is refused even when the file is really there.
        """
        content_type = STATIC_TYPES.get(Path(name).suffix.lower())
        if content_type is None:
            return error(404, "not found")
        try:
            path = safe_child(self.static_dir, name)
        except ValueError:
            return error(404, "not found")
        if not path.is_file():
            return error(404, "not found")
        return Response(200, path.read_bytes(), content_type)


def _redacted_event(event: dict[str, Any]) -> dict[str, Any]:
    """One event, with any credential-shaped text masked."""
    error_text = event.get("error")
    if not error_text:
        return event
    return {**event, "error": _redact(str(error_text))}


class Handler(BaseHTTPRequestHandler):
    """Thin adapter: read the request, ask the router, write it back."""

    server_version = "polyglot-webui"
    protocol_version = "HTTP/1.1"

    # The `do_*` names are fixed by BaseHTTPRequestHandler's dispatch table.
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        parts = urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            response = self.api.handle(
                method, parts.path, parse_qs(parts.query), body, dict(self.headers)
            )
        # A single bad request must not kill the thread that is serving it.
        except Exception:
            logger.exception("unhandled error for %s %s", method, parts.path)
            response = error(500, "internal error")
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if response.disposition:
            self.send_header("Content-Disposition", response.disposition)
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, fmt: str, *args) -> None:
        """Route the access log through logging instead of stderr noise."""
        logger.debug("%s - %s", self.address_string(), fmt % args)


def make_server(api: Api, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """A threaded server bound to loopback on *port*."""

    class _Handler(Handler):
        # `handle()` runs during BaseHTTPRequestHandler.__init__, so the
        # router has to be in place before super().__init__ runs.
        def __init__(self, *args, **kwargs) -> None:
            self.api = api
            super().__init__(*args, **kwargs)

    return ThreadingHTTPServer((HOST, port), _Handler)


def default_api(root: Path = REPO_ROOT, *, port: int = DEFAULT_PORT, **manager_kw) -> Api:
    """An ``Api`` wired to the repository's real data layout."""
    root = Path(root)
    return Api(
        root=root,
        manager=RunManager(root, **manager_kw),
        runs=RunsStore(root / LOGS_DIR),
        skills=SkillsStore(root / "skills"),
        glossaries=GlossaryStore(root / "glossaries"),
        port=port,
    )


def main(argv: list[str] | None = None) -> int:
    """Start the UI server and serve until interrupted."""
    parser = argparse.ArgumentParser(
        prog="python -m webui.server",
        description="Local web UI for polyglot-lc: start, watch and download runs.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port to bind on {HOST} (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root holding input/, logs/ and output/",
    )
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # A console on a legacy code page (cp1252) would raise on any non-ASCII
    # byte we print, and that must not be able to take the server down.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    api = default_api(root=args.root, port=args.port)
    server = make_server(api, args.port)
    print(f"polyglot-lc web UI on http://{HOST}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
