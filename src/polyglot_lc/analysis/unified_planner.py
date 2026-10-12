"""Unified document planner: summary + strategy + glossary + complexity in ONE call.

The previous head of the pipeline was sequential and expensive::

    summary tree (1 request per section + 1 for the document)
      -> plan_document (1 request)

For a 20-section document that is 21 Planner requests before the first chunk is
translated. :func:`analyze_document` folds all of it into a single structured
request, and degrades to a purely local heuristic plan when the model is
unavailable so the pipeline never depends on it succeeding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from polyglot_lc.analysis.complexity import heuristic_document_complexity
from polyglot_lc.segment.glossary import GlossaryTerm, glossary_to_md, render_user_glossary
from polyglot_lc.segment.summary_tree import build_sections, preview_text

if TYPE_CHECKING:
    from polyglot_lc.agents.base import Agent
    from polyglot_lc.segment.chunker import Chunk

#: Characters of the document body sent to the Planner (bounded context).
DOC_EXCERPT_CHARS = 3000
#: Characters of preview per section in the outline.
SECTION_PREVIEW_CHARS = 160
#: Hard cap on glossary pairs accepted from the Planner.
MAX_PLANNER_TERMS = 40

DEFAULT_STRATEGY = "literal, terminology-consistent"


@dataclass(frozen=True)
class GlossaryPair:
    """A source term and the target term it must be rendered as."""

    term: str
    translation: str
    reason: str = "planner"


@dataclass
class DocumentPlan:
    """Everything the per-chunk stage needs, produced in one Planner request."""

    summary: str = ""
    strategy: str = DEFAULT_STRATEGY
    few_shot: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    glossary_pairs: list[GlossaryPair] = field(default_factory=list)
    glossary_md: str = "(none)"
    complexity: float = 0.0
    section_notes: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    used_llm: bool = False
    section_count: int = 0

    def note_for(self, section_path: str) -> str:
        """Planner note for a chunk's section ('' when it has none)."""
        if not section_path or not self.section_notes:
            return ""
        if section_path in self.section_notes:
            return self.section_notes[section_path]
        # Section paths are "Parent > Child"; fall back to the parent heading.
        parent = section_path.split(" > ")[0].strip()
        return self.section_notes.get(parent, "")


def build_outline(chunks: list[Chunk]) -> dict[str, str]:
    """Deterministic section outline: section path -> preview (no LLM call)."""
    outline: dict[str, str] = {}
    for name, section_chunks in build_sections(chunks).items():
        outline[name] = preview_text(
            " ".join(c.text for c in section_chunks[:2]),
            limit=SECTION_PREVIEW_CHARS,
        )
    return outline


def structural_summary(chunks: list[Chunk], doc_title: str) -> str:
    """Local fallback summary built from headings alone (no LLM call)."""
    sections = [s for s in build_sections(chunks) if s and s != "(root)"]
    if not sections:
        return f"{doc_title}: {len(chunks)} chunk(s), no headings."
    listed = ", ".join(sections[:12])
    suffix = f" (+{len(sections) - 12} more)" if len(sections) > 12 else ""
    return f"{doc_title}: {len(chunks)} chunk(s) across sections: {listed}{suffix}."


def merge_glossary_md(
    *,
    planner_pairs: list[GlossaryPair],
    heuristic_terms: list[GlossaryTerm],
    user_glossary: dict[str, str] | None = None,
    extra_md: str = "",
) -> str:
    """Render the glossary block, most authoritative entry first.

    Order is deliberate: pre-rendered text supplied by the caller, then the
    user's CSV, then the Planner's translated terms, then whatever terminology
    heuristics found (which have no target form and are only hints).
    """
    covered: set[str] = set()
    blocks: list[str] = []

    if extra_md.strip():
        blocks.append(extra_md.strip())

    user = render_user_glossary(user_glossary or {})
    if user:
        blocks.append(user)
        covered.update(k.casefold() for k in (user_glossary or {}))

    pairs = [
        p
        for p in planner_pairs
        if p.term.strip() and p.translation.strip() and p.term.casefold() not in covered
    ]
    if pairs:
        blocks.append("\n".join(f"- {p.term} => {p.translation}" for p in pairs))
        covered.update(p.term.casefold() for p in pairs)

    leftovers = [t for t in heuristic_terms if t.term.casefold() not in covered]
    if leftovers:
        blocks.append(glossary_to_md(leftovers))

    return "\n".join(b for b in blocks if b and b != "(none)") or "(none)"


def _planner_prompt(
    *,
    doc_title: str,
    doc_text: str,
    doc_summary: str,
    outline: dict[str, str],
    src_lang: str,
    tgt_lang: str,
    domain: str,
    glossary_candidates: str,
    user_glossary_md: str,
    trace_lines: list[str],
) -> str:
    outline_md = "\n".join(f"- {name}: {preview}" for name, preview in outline.items())
    excerpt = doc_text[:DOC_EXCERPT_CHARS] if doc_text else doc_summary[:DOC_EXCERPT_CHARS]
    return (
        "You are a senior translation planner and terminologist.\n"
        f"Language pair: {src_lang} -> {tgt_lang}. Domain: {domain}.\n"
        f"Document: {doc_title}.\n\n"
        f"Section outline:\n{outline_md or '(no headings)'}\n\n"
        "Similar past runs from the reasoning bank:\n"
        + ("\n".join(trace_lines) if trace_lines else "(none)")
        + "\n\nTerminology candidates found by heuristics (keep, correct or drop):\n"
        + (glossary_candidates or "(none)")
        + "\n\nAuthoritative glossary supplied by the user (must be used verbatim):\n"
        + (user_glossary_md or "(none)")
        + f"\n\nDocument body (first {DOC_EXCERPT_CHARS} chars):\n{excerpt}\n\n"
        "Decide ONE translation strategy for this document, extract the domain "
        "terminology with its target-language form, and score how hard this "
        "document is to translate.\n"
        f"complexity is ONE number in [0, 1], not 0-10 and not a percentage: "
        "0.0 = plain prose in simple sentences, 0.5 = ordinary technical "
        "writing, 1.0 = dense terminology, heavy code or tables, long nested "
        "sentences. Use the whole range.\n"
        "Use section names from the outline above as the section_notes keys, "
        "copied verbatim (the parent heading name identifies a nested "
        "section).\n"
        "Reply ONLY with JSON:\n"
        '{"summary": "2-3 sentences", "strategy": "...", '
        '"few_shot": ["source => target", ...], "pitfalls": ["...", ...], '
        '"glossary_terms": [{"term": "...", "translation": "...", "reason": "..."}], '
        '"complexity": 0.0, "section_notes": {"<section name>": "short note"}, '
        '"notes": "..."}'
    )


def _fallback_plan(
    *,
    chunks: list[Chunk],
    doc_title: str,
    doc_text: str,
    outline: dict[str, str],
    heuristic_terms: list[GlossaryTerm],
    user_glossary: dict[str, str] | None,
) -> DocumentPlan:
    """Plan built locally when the Planner request fails or is disabled."""
    return DocumentPlan(
        summary=structural_summary(chunks, doc_title),
        strategy=DEFAULT_STRATEGY,
        glossary_pairs=[],
        glossary_md=merge_glossary_md(
            planner_pairs=[],
            heuristic_terms=heuristic_terms,
            user_glossary=user_glossary,
        ),
        complexity=heuristic_document_complexity(doc_text),
        section_count=len(outline),
        used_llm=False,
    )


def _parse_pairs(raw: Any) -> list[GlossaryPair]:
    pairs: list[GlossaryPair] = []
    seen: set[str] = set()
    if not isinstance(raw, list):
        return pairs
    for item in raw:
        if isinstance(item, dict):
            term = str(item.get("term", "")).strip()
            translation = str(item.get("translation", "")).strip()
            reason = str(item.get("reason", "planner")).strip() or "planner"
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            term, translation = str(item[0]).strip(), str(item[1]).strip()
            reason = "planner"
        else:
            continue
        if not term or not translation or term.casefold() in seen:
            continue
        seen.add(term.casefold())
        pairs.append(GlossaryPair(term=term, translation=translation, reason=reason))
        if len(pairs) >= MAX_PLANNER_TERMS:
            break
    return pairs


def _string_list(raw: Any, limit: int) -> list[str]:
    """Coerce a reply field to a list of strings, tolerating a bare string.

    A model that answers ``"few_shot": "a => b"`` instead of a one-element list
    would otherwise be iterated character by character.
    """
    if isinstance(raw, str):
        raw = [raw] if raw.strip() else []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x) for x in raw if str(x).strip()][:limit]


def _parse_section_notes(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    notes: dict[str, str] = {}
    for key, value in raw.items():
        text = str(value).strip()
        if text and str(key).strip():
            notes[str(key).strip()] = text
    return notes


def analyze_document(
    chunks: list[Chunk],
    planner: Agent,
    *,
    src_lang: str,
    tgt_lang: str,
    domain: str,
    doc_text: str = "",
    doc_summary: str = "",
    doc_title: str = "Document",
    heuristic_terms: list[GlossaryTerm] | None = None,
    user_glossary: dict[str, str] | None = None,
    rb_traces: list[Any] | None = None,
    extra_glossary_md: str = "",
) -> DocumentPlan:
    """Produce the whole document plan with **one** Planner request.

    Returns a :class:`DocumentPlan`; when the request fails the local heuristic
    plan is returned instead (``used_llm=False``) so translation still starts.
    """
    heuristic_terms = list(heuristic_terms or [])
    outline = build_outline(chunks)
    user_glossary_md = render_user_glossary(user_glossary or {})
    trace_lines = [
        f"- {t.strategy} | pitfalls: {'; '.join(t.pitfalls) or '-'}" for t in (rb_traces or [])
    ]
    prompt = _planner_prompt(
        doc_title=doc_title,
        doc_text=doc_text,
        doc_summary=doc_summary,
        outline=outline,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        domain=domain,
        glossary_candidates=glossary_to_md(heuristic_terms),
        user_glossary_md=user_glossary_md,
        trace_lines=trace_lines,
    )
    try:
        data = planner.chat_json([{"role": "user", "content": prompt}])
    except Exception as exc:  # noqa: BLE001 - planning must never block a run
        logger.warning("Unified planner unavailable ({}); using local plan", exc)
        plan = _fallback_plan(
            chunks=chunks,
            doc_title=doc_title,
            doc_text=doc_text,
            outline=outline,
            heuristic_terms=heuristic_terms,
            user_glossary=user_glossary,
        )
        if extra_glossary_md.strip():
            plan.glossary_md = merge_glossary_md(
                planner_pairs=[],
                heuristic_terms=heuristic_terms,
                user_glossary=user_glossary,
                extra_md=extra_glossary_md,
            )
        return plan

    pairs = _parse_pairs(data.get("glossary_terms"))
    try:
        complexity = min(1.0, max(0.0, float(data.get("complexity", 0.0))))
    except (TypeError, ValueError):
        complexity = heuristic_document_complexity(doc_text)
    summary = str(data.get("summary", "")).strip() or structural_summary(chunks, doc_title)
    return DocumentPlan(
        summary=summary,
        strategy=str(data.get("strategy", "")).strip() or DEFAULT_STRATEGY,
        few_shot=_string_list(data.get("few_shot"), 6),
        pitfalls=_string_list(data.get("pitfalls"), 6),
        glossary_pairs=pairs,
        glossary_md=merge_glossary_md(
            planner_pairs=pairs,
            heuristic_terms=heuristic_terms,
            user_glossary=user_glossary,
            extra_md=extra_glossary_md,
        ),
        complexity=complexity,
        section_notes=_parse_section_notes(data.get("section_notes")),
        notes=str(data.get("notes", "")),
        used_llm=True,
        section_count=len(outline),
    )
