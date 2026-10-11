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

from webui.paths import REPO_ROOT, safe_child
from webui.runs import LOGS_DIR, RunManager, RunPlanError
from webui.store import (
    EDITABLE_KEYS,
    READONLY_KEYS,
    ConfigStore,
    GlossaryStore,
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
        if path == "/api/config" and method == "GET":
            return self._config()
        if path.rstrip("/") == "/api/inputs" and method == "GET":
            return self._inputs()
        if path.startswith("/api/inputs/") and method == "PUT":
            # Unquote *before* validating: otherwise `..%2F.env` would look
            # like a harmless single-segment name and slip past `safe_child`.
            return self._upload(unquote(path[len("/api/inputs/"):]), query, body)
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
