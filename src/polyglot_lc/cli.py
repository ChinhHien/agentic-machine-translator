"""polyglot-lc CLI entry point (typer).

Output is intentionally simple: one line per running pipeline step (printed by
the pipeline's :class:`~polyglot_lc.ui.simple.SimpleUI`) followed by a compact
token/request report. No dynamic dashboard, spinner, progress bar or live
redraw. Full structured logs still go to ``logs/run_<ts>.jsonl``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from polyglot_lc import __version__
from polyglot_lc.config import get_app_config
from polyglot_lc.llm.client import TokenUsage, provider_gate_stats
from polyglot_lc.llm.resilience import RetryStats
from polyglot_lc.skillopt.skill_store import SKILL_SCHEMA, SkillStore

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


def _skill_summary(result: object) -> str:
    """The skill the run used, what it measured, and whether it was replaced."""
    if not getattr(result, "skill_applied", False):
        return "không dùng skill (SKILL_APPLY=off hoặc chưa có skill)"
    mean = (
        f"đo {result.skill_mean:.2f}/10 ({result.skill_samples} chunk)"
        if getattr(result, "skill_samples", 0)
        else "chưa đo"
    )
    promoted = "có" if getattr(result, "skill_promoted", False) else "không"
    return (
        f"v{result.skill_version} ({result.skill_id}) · {mean}"
        f" · promote: {promoted}"
    )


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
    # Rich's legacy Windows renderer writes through the stream's own codec,
    # and it is chosen at Console construction from the environment — a
    # redirected run on Windows still hits it. A cp1252 console would then
    # raise on a Vietnamese glyph in the final report, *after* the output
    # files are written, so the exit code lied about an otherwise good run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # a stream that cannot be reconfigured is not worth a crash


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
        reported = hits + misses
        lines.append("")
        lines.append("Cache (per request):")
        lines.append(f"  hits:    {hits:,}")
        lines.append(f"  misses:  {misses:,}")
        lines.append(f"  unknown: {unknown:,}")
        lines.append(f"  hit rate: {100 * hits // reported}% of {reported:,} reported request(s)")
        by_agent = TokenUsage.snapshot_by_agent()
        per_agent = [
            f"{name} {b['cache_hits']}/{b['cache_hits'] + b['cache_misses']}"
            for name, b in by_agent.items()
            if b.get("cache_hits") or b.get("cache_misses")
        ]
        if len(per_agent) > 1:
            # Which agent's stable prefix the provider is actually caching: a
            # Translator at 0% has a prefix that changes every chunk.
            lines.append("  by agent (hits/reported): " + ", ".join(per_agent))

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

    moving = {key: s for key, s in provider_gate_stats().items() if s["adaptive"] or s["throttles"]}
    if moving:
        # Only worth a line when the queue actually moved (or was pushed back):
        # a static limit is just LLM_PROVIDER_CONCURRENCY, which the caller set.
        lines.append("")
        lines.append("Provider slots (concurrency):")
        for key, s in moving.items():
            extra = f", {s['throttles']} throttle backoff(s)" if s["throttles"] else ""
            lines.append(
                f"  {key}: {s['start']} -> {s['limit']} of {s['ceiling']} "
                f"(peak {s['peak']} in flight{extra})"
            )
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
    reuse_foreign: Annotated[
        bool,
        typer.Option(
            "--reuse-foreign/--no-reuse-foreign",
            help="On --resume, reuse a verified translation of the same chunk text "
            "even when it was made by another model or endpoint (default: on). "
            "Pass --no-reuse-foreign to translate every chunk again.",
        ),
    ] = True,
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
            src="en",
            tgt="vi",
            out=str(out) if out else None,
            resume=resume,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
            output_format=output_format,
            reuse_foreign=reuse_foreign,
        )
    except Exception as exc:
        console.print(f"Run failed: {type(exc).__name__}: {exc}", markup=False)
        raise typer.Exit(code=1) from exc

    resilience = RetryStats.snapshot()
    if result.chunks_failed or result.fatal_error:
        # The verdict is on the deliverable: a chunk never got its translation,
        # or the run ended on an error.
        outcome = "Failed"
    elif result.incomplete_chunks:
        # Every chunk may have words, but some never got a verified checkpoint.
        # Calling that "completed" is the false report this run exists to end.
        outcome = "Incomplete"
    elif result.errors:
        # Everything asked for was delivered. The provider failures the run
        # waited out are listed below, but the file itself is complete.
        outcome = "Completed with errors"
    else:
        outcome = "Completed"
    critic_label = "avg score (est.)" if result.avg_critic_estimate else "avg score"
    console.print("", markup=False)
    console.print(f"{outcome}.", markup=False)
    console.print(
        f"  chunks: {result.chunks}  {critic_label}: {result.avg_critic:.2f}/10"
        f"  retries: {result.retries}  traces: {result.traces_stored}",
        markup=False,
    )
    if result.incomplete_chunks:
        console.print(
            f"  completion: {result.verified_chunks}/{result.expected_chunks} chunk có "
            f"checkpoint đã xác minh; còn {len(result.incomplete_chunks)} chunk chưa xong",
            markup=False,
        )
        for item in result.incomplete_chunks[:20]:
            console.print(
                f"    - {item.get('chunk', '')}: {item.get('reason', '')}",
                markup=False,
            )
        if len(result.incomplete_chunks) > 20:
            console.print(
                f"    ... và {len(result.incomplete_chunks) - 20} chunk khác (xem log)",
                markup=False,
            )
    cumulative = result.cumulative or {}
    attempts = int(cumulative.get("attempts", 0) or 0)
    if attempts > 1:
        tokens = cumulative.get("tokens") or {}
        skill = cumulative.get("skill") or {}
        duration_s = int(cumulative.get("duration_ms", 0) or 0) / 1000.0
        console.print(
            f"  total after {attempts} attempts: "
            f"input {int(tokens.get('sent', 0) or 0):,} / "
            f"output {int(tokens.get('received', 0) or 0):,} tokens, "
            f"retries {int(cumulative.get('retries', 0) or 0)}, "
            f"planner requests {int(cumulative.get('planner_calls', 0) or 0)}, "
            f"{duration_s:.1f}s, skill {float(skill.get('mean', 0) or 0):.2f}/10 "
            f"({int(skill.get('samples', 0) or 0)} chunk)",
            markup=False,
        )
    console.print(f"  skill: {_skill_summary(result)}", markup=False)
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
    if result.reused_foreign:
        # The words are there, but they are another model's: saying so keeps a
        # mixed document from reading like a single model's work.
        console.print(
            f"  reused from another model: {result.reused_foreign} chunk(s)",
            markup=False,
        )
    if result.consistency_patch_budget:
        console.print(
            f"  consistency patch budget: {result.consistency_patch_budget} attempt(s), "
            f"{result.consistency_patches} applied; semantic verification costs extra requests",
            markup=False,
        )
    if result.consistency_skipped or result.consistency_rejected:
        # The document is *not* fully consistent: the pass ran out of budget, or
        # refused rewrites that would have cost the translation. Saying so is
        # the point of the count.
        console.print(
            f"  consistency left {result.consistency_skipped} chunk(s) drifting"
            f"  patch rejected: {result.consistency_rejected}",
            markup=False,
        )
    if result.consistency_stopped_early:
        # The pass gave up on the editor, not on the document: its remaining
        # budget went unspent and the untouched chunks are already counted.
        console.print(
            "  consistency: stopped after two patches in a row were refused; "
            "remaining budget unspent",
            markup=False,
        )
    if result.consistency_label_drift:
        console.print(
            f"  consistency: {result.consistency_label_drift} label(s) drifted"
            f" out of {result.consistency_labels} compared",
            markup=False,
        )
    if result.consistency_no_pairs:
        # "0 inconsistencies" over zero terms is not a clean document: the
        # Planner fell back (or no glossary existed), so only labels were
        # compared. Saying which one it was keeps the two from reading alike.
        console.print(
            "  consistency: no glossary term to compare - only labels were checked",
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
    for review in result.quality_review:
        # The document stays clean: what a human reviewer needs is here and in
        # the JSONL log, not inside the translation.
        console.print(
            f"  review {review.get('chunk', '')}: "
            f"{review.get('severity', '')} — {review.get('reason', '')[:160]}",
            markup=False,
        )

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
    missing = result.chunks_failed or len(result.incomplete_chunks)
    if missing:
        # A permanently failed chunk has no translation but is counted as
        # incomplete, not as a failed run, so the count comes from the verdict.
        console.print(
            f"incomplete: {missing}/{result.chunks} chunk(s) missing; "
            "use --resume to retry only unfinished chunks",
            markup=False,
        )
    if result.chunks_failed or result.fatal_error:
        raise typer.Exit(code=1)
    if result.incomplete_chunks:
        # 2, not 1: the file exists, but this is not a finished translation.
        # A supervisor can tell "crashed" from "needs another pass" by this.
        raise typer.Exit(code=2)


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


@app.command()
def skills(
    history: Annotated[
        int,
        typer.Option("--history", "-H", min=0, help="In N dòng lịch sử gần nhất (0 = không)."),
    ] = 0,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Kiểm tra schema và nội dung skill.md rồi thoát."),
    ] = False,
) -> None:
    """Skill duy nhất của hệ thống: nội dung, lần đo và lịch sử.

    Không có skill nào thì nói vậy, không phải lỗi: SkillOpt chỉ ghi khi một
    lượt dịch đã chạy đủ mẫu và vượt baseline.
    """
    store = SkillStore(get_app_config().skills_dir)
    if validate:
        issues = store.validate()
        if not issues:
            console.print(f"skill.md hợp lệ (schema {SKILL_SCHEMA})", markup=False)
            return
        console.print(f"{len(issues)} vấn đề", markup=False)
        for issue in issues:
            console.print(f"  - {issue}", markup=False)
        raise typer.Exit(code=1)

    skill = store.load()
    if skill is None:
        console.print("chưa có skill (chưa promote hoặc file không đọc được)", markup=False)
    else:
        measured = (
            f"đo {skill['mean_score']:.2f}/10 khai {skill['measured_runs']} lượt"
            if skill.get("mean_score") is not None
            else "chưa đo (lượt tới dùng skill này sẽ đo)"
        )
        promoted_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(skill.get("promoted_at", 0) or 0)))
        console.print(
            f"system v{skill['version']} ({skill['skill_id']}) · scope {skill['scope']}",
            markup=False,
        )
        console.print(
            f"  {measured} · baseline {float(skill.get('baseline_mean', 0.0)):.2f}/10"
            f" · {len(skill.get('few_shot') or [])} few-shot",
            markup=False,
        )
        console.print(
            f"  promote: {promoted_at} · từ run {skill.get('promoted_from_run') or '—'}",
            markup=False,
        )
        console.print(f"  file: {store.path}", markup=False)

    if history:
        rows = store.history(limit=history)
        if not rows:
            console.print("  lịch sử: chưa có lượt nào ghi lại", markup=False)
            return
        console.print("  lịch sử (cũ → mới):", markup=False)
        for row in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(row.get("at", 0) or 0)))
            mean = row.get("mean")
            measured = f"{mean:.2f}" if isinstance(mean, (int, float)) else "—"
            verdict = "promote" if row.get("promoted") else str(row.get("reason") or "giữ")
            console.print(
                f"    {when}  v{row.get('version') or '?'}  n={row.get('n')}  mean={measured}"
                f"  {verdict}  {row.get('doc') or row.get('run_id') or ''}",
                markup=False,
            )


if __name__ == "__main__":
    app()
