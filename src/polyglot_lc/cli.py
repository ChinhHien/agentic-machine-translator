"""polyglot-lc CLI entry point (typer).

Output is intentionally simple: one line per running pipeline step (printed by
the pipeline's :class:`~polyglot_lc.ui.simple.SimpleUI`) followed by a compact
token/request report. No dynamic dashboard, spinner, progress bar or live
redraw. Full structured logs still go to ``logs/run_<ts>.jsonl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"polyglot-lc {__version__}", markup=False)
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
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
    file: Path = typer.Option(
        ..., "--file", "-f", exists=True, dir_okay=False, readable=True,
        help="Input file (.md, .txt, .pdf).",
    ),
    src: str = typer.Option(
        "auto", "--src", "-s", help="Source language ('auto' = langdetect)."
    ),
    tgt: str = typer.Option(
        "vi", "--tgt", "-t", help="Target language code."
    ),
    domain: str = typer.Option(
        "general", "--domain", "-d", help="Domain hint (tech, legal, ...)."
    ),
    glossary: Optional[Path] = typer.Option(
        None, "--glossary", "-g", exists=True, dir_okay=False,
        help="CSV glossary (src,tgt per row).",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output file path."
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-m", min=64, help="Chunk token budget."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only print the final report (no step lines)."
    ),
) -> None:
    """Translate FILE with the multi-agent pipeline."""
    from polyglot_lc.pipeline import PipelineResult, TranslationPipeline

    try:
        pipeline = TranslationPipeline(quiet=quiet)
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
        )
    except Exception as exc:
        console.print(f"Run failed: {type(exc).__name__}: {exc}", markup=False)
        raise typer.Exit(code=1) from exc

    resilience = RetryStats.snapshot()
    outcome = "Completed" if not result.errors else "Completed with errors"
    console.print("", markup=False)
    console.print(f"{outcome}.", markup=False)
    console.print(
        f"  chunks: {result.chunks}  avg critic: {result.avg_critic:.2f}/10"
        f"  retries: {result.retries}  traces: {result.traces_stored}"
        f"  skill promoted: {result.skill_promoted}",
        markup=False,
    )
    if resilience["fallbacks"] or resilience["rate_limit_retries"] \
            or resilience["transient_retries"]:
        console.print(
            f"  resilience: {resilience['rate_limit_retries']} rate-limit retry(ies), "
            f"{resilience['transient_retries']} transient retry(ies), "
            f"{resilience['fallbacks']} fallback(s)",
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
    if result.log_path:
        console.print(f"log file    : {result.log_path}", markup=False)


if __name__ == "__main__":
    app()
