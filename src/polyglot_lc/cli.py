"""polyglot-lc CLI entry point (typer).

Output is intentionally simple: one line per running pipeline step (printed by
the pipeline's :class:`~polyglot_lc.ui.simple.SimpleUI`) followed by a compact
token/request report. No dynamic dashboard, spinner, progress bar or live
redraw. Full structured logs still go to ``logs/run_<ts>.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from polyglot_lc import __version__
from polyglot_lc.llm.client import TokenUsage
from polyglot_lc.llm.resilience import RetryStats

app = typer.Typer(
    name="polyglot",
    help="Long-context multi-agent machine translation.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(highlight=False)


def _console_safe(text: str, encoding: str | None = None) -> str:
    """Text this console can actually print.

    A legacy Windows console (cp1252) cannot encode every character a trace
    legitimately contains — a ``⟦CODE_n⟧`` marker, a Vietnamese filename — and
    Rich's legacy renderer raises rather than replacing. Encoding to the
    console's own codec with ``replace`` keeps the command readable instead of
    printing a traceback about one glyph.
    """
    codec = encoding or getattr(console.file, "encoding", None) or "utf-8"
    try:
        text.encode(codec)
    except (UnicodeEncodeError, LookupError):
        return text.encode(codec, "replace").decode(codec, "replace")
    return text


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"polyglot-lc {__version__}", markup=False)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """polyglot-lc: multi-agent long-context translation."""


def _token_report() -> str:
    """Compact, provider-accurate token/request report for the whole run."""
    usage = TokenUsage.snapshot()
    sent = usage["sent"]
    received = usage["received"]
    cached = usage.get("cached", 0)
    uncached = max(0, sent - cached)
    lines = ["Token usage:"]
    if cached:
        lines.append(f"  Input:          {sent:,}")
        lines.append(f"  Cached input:   {cached:,}")
        lines.append(f"  Uncached input: {uncached:,}")
    else:
        lines.append(f"  Input:  {sent:,}")
    lines.append(f"  Output: {received:,}")
    lines.append(f"  Total:  {sent + received:,}")
    lines.append(f"  Requests: {usage['calls']:,}")
    if usage.get("estimated_calls"):
        lines.append(
            f"  (estimated usage for {usage['estimated_calls']:,} request(s): "
            "provider returned no usage metadata)"
        )

    hits = usage.get("cache_hits", 0)
    misses = usage.get("cache_misses", 0)
    unknown = usage.get("cache_unknown", 0)
    if hits or misses:
        lines.append("")
        lines.append("Cache (per request):")
        lines.append(f"  hits:    {hits:,}")
        lines.append(f"  misses:  {misses:,}")
        lines.append(f"  unknown: {unknown:,}")

    by_model = TokenUsage.snapshot_by_model()
    if len(by_model) > 1:
        lines.append("")
        lines.append("By provider / model:")
        for key, b in by_model.items():
            detail = (
                f"input {b['sent']:,}"
                + (f" (cached {b['cached']:,})" if b.get("cached") else "")
                + f", output {b['received']:,}, {b['calls']:,} request(s)"
            )
            lines.append(f"  {key}")
            lines.append(f"    {detail}")
    return "\n".join(lines)


@app.command()
def translate(
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input PDF document.",
        ),
    ],
    src: Annotated[
        str, typer.Option("--src", "-s", help="Source language ('auto' = langdetect).")
    ] = "auto",
    tgt: Annotated[str, typer.Option("--tgt", "-t", help="Target language code.")] = "vi",
    domain: Annotated[
        str, typer.Option("--domain", "-d", help="Domain hint (tech, legal, ...).")
    ] = "general",
    glossary: Annotated[
        Path | None,
        typer.Option(
            "--glossary",
            "-g",
            exists=True,
            dir_okay=False,
            help="CSV glossary (src,tgt per row).",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output base path; `<base>.md` and/or `<base>.pdf` are written.",
        ),
    ] = None,
    output_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-F",
            help="What to write: md (default), pdf (source PDF rebuilt with the "
            "translation) or both (env: OUTPUT_FORMAT).",
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", "-m", min=64, help="Chunk token budget."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only print the final report (no step lines)."),
    ] = False,
    resume: Annotated[
        bool | None,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse per-chunk checkpoints from an interrupted run "
            "(default: RESUME env var, off).",
        ),
    ] = None,
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-dir",
            dir_okay=True,
            file_okay=False,
            help="Directory for per-chunk checkpoints (default: CHECKPOINT_DIR).",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Name of this run (default: RUN_ID from the environment, else a new id). "
            "Passing a known id is how a trace is later tied to a launcher's own record.",
        ),
    ] = None,
) -> None:
    """Translate FILE with the multi-agent pipeline."""
    from polyglot_lc.pipeline import PipelineResult, TranslationPipeline

    try:
        pipeline = TranslationPipeline(quiet=quiet, run_id=run_id)
    except Exception as exc:  # ConfigError and friends
        console.print(
            f"Configuration error: {type(exc).__name__}: {exc}\n\n"
            "Copy .env.example to .env and fill in the agent settings "
            "(or set LLM_MOCK=1 for a keyless demo).",
            markup=False,
        )
        raise typer.Exit(code=2) from exc

    try:
        result: PipelineResult = pipeline.run(
            file_path=file,
            src=src,
            tgt=tgt,
            domain=domain,
            glossary=str(glossary) if glossary else None,
            out=str(out) if out else None,
            max_tokens=max_tokens,
            resume=resume,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
            output_format=output_format,
        )
    except Exception as exc:
        console.print(f"Run failed: {type(exc).__name__}: {exc}", markup=False)
        raise typer.Exit(code=1) from exc

    resilience = RetryStats.snapshot()
    if result.chunks_failed or result.fatal_error:
        # The verdict is on the deliverable: a chunk never got its translation,
        # or the run ended on an error.
        outcome = "Failed"
    elif result.errors:
        # Everything asked for was delivered. The provider failures the run
        # waited out are listed below, but the file itself is complete.
        outcome = "Completed with errors"
    else:
        outcome = "Completed"
    critic_label = "avg critic (est.)" if result.avg_critic_estimate else "avg critic"
    console.print("", markup=False)
    console.print(f"{outcome}.", markup=False)
    console.print(
        f"  chunks: {result.chunks}  {critic_label}: {result.avg_critic:.2f}/10"
        f"  retries: {result.retries}  traces: {result.traces_stored}"
        f"  skill promoted: {result.skill_promoted}",
        markup=False,
    )
    console.print(
        f"  routes: fast {result.fast_path_chunks} / standard "
        f"{result.standard_path_chunks} / deep {result.deep_path_chunks}"
        f"  planner requests: {result.planner_calls}"
        f"  glossary terms: {result.glossary_terms}",
        markup=False,
    )
    if result.fast_path_rejections:
        # Why the fast path was refused, from the run's own decisions: the answer
        # to "why did this chunk cost four requests?" without opening the trace.
        console.print(
            f"  fast path refused for {result.fast_path_rejections} chunk(s); "
            f"most often: {result.fast_path_top_reason or 'unrecorded'}",
            markup=False,
        )
    if result.resumed_chunks or result.consistency_patches:
        console.print(
            f"  resumed chunks: {result.resumed_chunks}"
            f"  consistency patches: {result.consistency_patches}"
            f"  elapsed: {result.elapsed_s:.1f}s",
            markup=False,
        )
    if (
        resilience["fallbacks"]
        or resilience["rate_limit_retries"]
        or resilience["transient_retries"]
        or resilience["chunk_retries"]
    ):
        console.print(
            f"  resilience: {resilience['rate_limit_retries']} rate-limit retry(ies), "
            f"{resilience['transient_retries']} transient retry(ies), "
            f"{resilience['fallbacks']} fallback(s), "
            f"{resilience['chunk_retries']} chunk run again after errors",
            markup=False,
        )
    if result.errors:
        console.print("  errors:", markup=False)
        for err in result.errors[-5:]:
            console.print(f"    - {err}", markup=False)

    console.print("", markup=False)
    console.print(_token_report(), markup=False)
    console.print("", markup=False)
    console.print(f"output file : {result.output_path}", markup=False)
    if result.pdf_path and result.pdf_path != result.output_path:
        console.print(
            f"pdf file    : {result.pdf_path}  ({result.pdf_blocks} block(s) rebuilt)",
            markup=False,
        )
    if result.log_path:
        console.print(f"log file    : {result.log_path}", markup=False)


@app.command()
def trace(
    target: Annotated[
        str | None,
        typer.Argument(
            help="A run id, a log file, or nothing: the newest run under the log directory."
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Print the same view as machine-readable JSON."),
    ] = False,
    log_dir: Annotated[
        Path | None,
        typer.Option(
            "--log-dir",
            dir_okay=True,
            file_okay=False,
            help="Where to look for runs (default: the parent of LOGS_DIR, i.e. logs/).",
        ),
    ] = None,
) -> None:
    """Explain one finished run: what it was asked to do, how it went, what it cost.

    Reading a run that broke is ordinary, not a failure of this command: a run
    that died before its first record exits 0 and says so. Only "there is no run
    to read" is an error.
    """
    from polyglot_lc.ui.trace import default_root, read_trace, resolve_run

    root = log_dir if log_dir is not None else default_root()
    try:
        path = resolve_run(target, root)
    except FileNotFoundError as exc:
        console.print(f"No run to read: {exc}", markup=False)
        raise typer.Exit(code=1) from exc

    view = read_trace(path)
    if json_out:
        # ``soft_wrap`` because Rich otherwise soft-wraps at the console width,
        # which inserts newlines *inside* JSON strings and makes the output
        # unparseable. Both branches print text that must survive verbatim.
        console.print(
            _console_safe(json.dumps(view.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)),
            markup=False,
            soft_wrap=True,
        )
    else:
        console.print(_console_safe(view.as_text()), markup=False, soft_wrap=True)


if __name__ == "__main__":
    app()
