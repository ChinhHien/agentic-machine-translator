"""HTTP surface for the web UI: a pure router plus a thin socket handler.

``Api.handle`` takes plain data and returns plain data, so nearly every test
can run without opening a port; :class:`Handler` only reads a request, calls
the router and writes the response back. The server binds ``127.0.0.1`` on
purpose: it can start and kill translation runs, so it must not be reachable
from the network.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from webui.paths import REPO_ROOT, is_within, safe_child
from webui.runs import LOGS_DIR, RunManager, RunPlanError
from webui.store import (
    EDITABLE_KEYS,
    READONLY_KEYS,
    ConfigStore,
    GlossaryStore,
    RunRecord,
    RunsStore,
    SkillsStore,
)

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

#: Markdown, for the one route that returns a translated document.
MARKDOWN_TYPE = "text/markdown; charset=utf-8"

#: Fields the run list shows; the command line and pid stay in the detail view.
RUN_SUMMARY_FIELDS = (
    "run_id", "started", "status", "file", "src", "tgt", "domain", "chunks",
    "exit_code", "output_path", "mock", "note",
)


@dataclass
class Response:
    """One HTTP response, in the plainest possible shape."""

    status: int
    body: bytes
    content_type: str


def json_response(payload: object, status: int = 200) -> Response:
    """A JSON response; non-ASCII text stays readable instead of escaped."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return Response(status, body, JSON_TYPE)


def error(status: int, message: str) -> Response:
    """A JSON error, so the front end never has to parse HTML for a failure."""
    return json_response({"error": message}, status)


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


def _run_summary(record: RunRecord) -> dict[str, Any]:
    """The subset of a record the run list shows, in a stable key order."""
    data = record.to_dict()
    return {field: data[field] for field in RUN_SUMMARY_FIELDS}


class Api:
    """Routes requests to the stores and the run manager."""

    def __init__(
        self,
        root: Path,
        manager: RunManager,
        runs: RunsStore,
        skills: SkillsStore,
        glossaries: GlossaryStore,
        config: ConfigStore,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.root = Path(root)
        self.manager = manager
        self.runs = runs
        self.skills = skills
        self.glossaries = glossaries
        self.config = config
        self.port = port

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
            return self._static(path[len("/static/"):])
        if path == "/api/health" and method == "GET":
            return json_response({"ok": True})
        return self._api(method, path, query, body, headers)

    def _api(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: bytes,
        headers: dict[str, str],
    ) -> Response:
        """API routes; a method/path pair we do not serve is a JSON 404."""
        if path == "/api/config":
            if method == "GET":
                return self._config()
            if method == "PUT":
                return self._put_config(body)
        if path.rstrip("/") == "/api/runs":
            if method == "GET":
                return self._runs()
            if method == "POST":
                return self._start_run(body)
        if path.startswith("/api/runs/"):
            answer = self._run_route(method, path[len("/api/runs/"):], query)
            if answer is not None:
                return answer
        if path.rstrip("/") == "/api/inputs" and method == "GET":
            return self._inputs()
        if path.startswith("/api/inputs/") and method == "PUT":
            # Unquote *before* validating: otherwise `..%2F.env` would look
            # like a harmless single-segment name and slip past `safe_child`.
            return self._upload(unquote(path[len("/api/inputs/"):]), query, body)
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
        return frozenset({
            f"http://127.0.0.1:{self.port}",
            f"http://localhost:{self.port}",
        })

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

    # Config ---------------------------------------------------------------
    def _put_config(self, body: bytes) -> Response:
        """Write the settings the UI is allowed to change; reject the rest.

        Credentials are not merely refused here — ``ConfigStore.save`` never
        rewrites a line it was not asked about, so a key in `.env` survives
        untouched even though this route writes the same file.
        """
        data = _json_object(body)
        if data is None:
            return error(400, "body must be a JSON object")
        values = data.get("values")
        if not isinstance(values, dict):
            return error(400, "values must be an object of settings")
        saved, rejected = self.config.save(values)
        return json_response({"saved": saved, "rejected": rejected})

    def _config(self) -> Response:
        """Effective settings, the whitelist, and which credentials are set.

        Values come from the probe, but are then filtered through the
        whitelist: the server must not be able to leak a credential even if the
        probe (or a future version of it) starts reporting one.
        """
        payload: dict[str, Any] = {
            "values": {},
            "sources": {},
            "editable": list(EDITABLE_KEYS),
            "readonly": list(READONLY_KEYS),
            "secrets": self.config.secret_status(),
        }
        try:
            probed = self.manager.probe(["--config"])
        except RunPlanError as exc:
            # Not fatal: the settings screen still works and can say why the
            # effective values are missing. Without a reason the user is stuck.
            payload["probe_error"] = str(exc)
            return json_response(payload)
        allowed = set(EDITABLE_KEYS) | set(READONLY_KEYS)
        for field in ("values", "sources"):
            reported = probed.get(field)
            if isinstance(reported, dict):
                payload[field] = {k: v for k, v in reported.items() if k in allowed}
        return json_response(payload)

    # Inputs ---------------------------------------------------------------
    def _inputs(self) -> Response:
        """Documents available to translate, newest metadata included."""
        directory = self.root / INPUT_DIR
        files = []
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    info = path.stat()
                    files.append({"name": path.name, "size": info.st_size,
                                  "mtime": info.st_mtime})
        return json_response({"files": files})

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
            summaries.append(_run_summary(record))
        return json_response({"runs": summaries})

    def _run_route(self, method: str, rest: str, query: dict[str, list[str]]) -> Response | None:
        """``/api/runs/<id>`` and its sub-resources; ``None`` when unknown."""
        parts = [unquote(part) for part in rest.split("/")]
        run_id = parts[0]
        if len(parts) == 1 and method == "GET":
            return self._run_detail(run_id)
        if len(parts) == 2 and method == "GET" and parts[1] == "events":
            return self._run_events(run_id, _first(query, "after"))
        if len(parts) == 2 and method == "GET" and parts[1] == "output":
            return self._run_output(run_id)
        if len(parts) == 2 and method == "POST" and parts[1] == "stop":
            return self._stop_run(run_id)
        return None

    def _run_detail(self, run_id: str) -> Response:
        try:
            record = self.manager.status(run_id)
        except KeyError:
            return error(404, f"no such run: {run_id}")
        return json_response({"run": record.to_dict()})

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
        return json_response({
            "events": events,
            "next": next_index,
            "status": record.status,
            "exit_code": record.exit_code,
            "malformed": malformed,
            # Without this the UI cannot tell "nothing has happened yet" from
            # "this run will never report anything", and would sit silent.
            "has_events": self.manager.has_events(run_id),
        })

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
                mock=bool(data.get("mock")),
            )
        except RunPlanError as exc:
            # The document could not be read or planned: nothing was started.
            return error(422, str(exc))
        except ValueError as exc:
            return error(403, f"refused: {exc}")
        return json_response({
            "run_id": record.run_id,
            "chunks": record.chunks,
            "output_path": record.output_path,
            "command": record.command,
        }, 201)

    def _stop_run(self, run_id: str) -> Response:
        """Stop a run; the answer carries the state it actually reached."""
        try:
            record = self.manager.stop(run_id)
        except KeyError:
            return error(404, f"no such run: {run_id}")
        except RuntimeError as exc:
            return error(409, f"cannot stop: {exc}")
        return json_response({"status": record.status, "exit_code": record.exit_code})

    def _run_output(self, run_id: str) -> Response:
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
        path = Path(record.output_path)
        if not is_within(self.root, path):
            return error(403, "refused: output path is outside the repository")
        if not path.is_file():
            return error(404, "this run has not written its output yet")
        return Response(200, path.read_bytes(), MARKDOWN_TYPE)

    # Glossaries and skills -------------------------------------------------
    def _glossary_route(self, method: str, path: str, body: bytes) -> Response | None:
        if path.rstrip("/") == "/api/glossaries" and method == "GET":
            return json_response({"files": self.glossaries.list_files()})
        if not path.startswith("/api/glossaries/"):
            return None
        name = unquote(path[len("/api/glossaries/"):])
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
        parts = [unquote(part) for part in path[len("/api/skills/"):].split("/")]
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

    def _upload(self, name: str, query: dict[str, list[str]], body: bytes) -> Response:
        """Write a document into ``input/``, refusing to clobber one by accident."""
        try:
            path = safe_child(self.root / INPUT_DIR, name)
        except ValueError:
            return error(403, f"refused: {name!r} is not a plain file name")
        if path.exists() and not _flagged(query, "overwrite"):
            return error(409, f"{name} already exists; pass ?overwrite=1 to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return json_response({"name": name, "size": len(body)})

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
        config=ConfigStore(root, root / ".env"),
        port=port,
    )


def main(argv: list[str] | None = None) -> int:
    """Start the UI server and serve until interrupted."""
    parser = argparse.ArgumentParser(
        prog="python -m webui.server",
        description="Local web UI for polyglot-lc: start, watch and stop runs.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to bind on {HOST} (default {DEFAULT_PORT})")
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root holding input/, logs/ and output/")
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

    api = default_api(args.root, port=args.port)
    server = make_server(api, args.port)
    host, port = server.server_address[:2]
    sys.stdout.write(f"polyglot-lc web UI on http://{host}:{port}  (Ctrl+C to stop)\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stdout.write("\nstopping\n")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
