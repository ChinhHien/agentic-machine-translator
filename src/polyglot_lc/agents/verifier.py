"""Verifier agent: final hard gates (placeholders, length, JSON verdict)."""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from typing import Any

from loguru import logger

from polyglot_lc.agents.base import Agent
from polyglot_lc.llm.client import parse_json_reply
from polyglot_lc.llm.resilience import RAISED_OUTPUT_CAP, LLMError, TruncatedError

_PH_RE = re.compile(r"⟦CODE_\d+⟧")

_SEVERITY_ORDER = {"minor": 0, "major": 1, "fatal": 2}

#: Environment knob for the verifier's output budget.
_VERIFIER_MAX_TOKENS_ENV = "VERIFIER_MAX_TOKENS"

#: Hard ceiling for the recovery retry. Doubling is unbounded otherwise: a cap
#: raised through ``VERIFIER_MAX_TOKENS`` would ask for twice itself on the
#: retry, and gateways reject a ``max_tokens`` above their own limit outright.
#:
#: Was 8192, which a real run reached and cut at three times: each of those
#: verdicts then degraded to a *minimal* one that checks less than the full
#: verdict. The same endpoint served translator replies of ~17000 completion
#: tokens with no cap set, so it demonstrably allows more than 8192 — the
#: ceiling now matches the Translator's, since both ask the same model.
VERIFIER_MAX_TOKENS_CEILING = 32768

#: Output caps live in the shared capability registry, keyed ``(base_url, model)``
#: and persisted. A reasoning model that cut one verdict at 4096 will cut the next
#: chunk the same way, so the pipeline starts those calls at the raised cap
#: instead of paying for the same failed attempt once per chunk — and the
#: Translator asking the same endpoint benefits from what the Verifier learned.
#: These three functions are the thin view the Verifier keeps over that store.


def _cap_for(provider: str, model: str) -> int:
    """The output budget this call starts with."""
    return _verifier_output_cap()


def _reasoning_kwargs() -> dict[str, Any]:
    """Provider hint that keeps the output budget for the answer, when asked.

    ``VERIFIER_REASONING=off`` sends the hint reasoning gateways understand
    (``thinking.type = disabled``). Support is the gateway's business: an
    endpoint that ignores it keeps reasoning, which costs nothing here; one
    that rejects the field fails loudly rather than silently truncating. The
    default, ``auto``, sends nothing at all.
    """
    raw = os.getenv("VERIFIER_REASONING", "").strip().lower()
    if raw in {"off", "none", "disabled", "disable"}:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _verdict_from_partial(err: TruncatedError) -> dict[str, Any] | None:
    """The verdict inside a reply flagged as cut, when the answer is complete.

    A reasoning model can finish its JSON and then keep thinking until the cap
    runs out; ``finish_reason`` says ``length`` while the content already holds
    a whole verdict. Only a parseable JSON object counts — anything else is a
    real truncation.
    """
    text = getattr(err, "partial_text", "") or ""
    if not text.strip():
        return None
    try:
        data = parse_json_reply(text)
    except LLMError:
        return None
    logger.info("VERIFIER reply flagged as cut, but its JSON is complete; using it")
    return data


def _verifier_output_cap() -> int:
    """Output-token budget for one verifier call, in tokens.

    Gateways default to 2048 completion tokens when the request sets no cap,
    and a verbose verdict (per-check reasons for a 700-token chunk) overruns
    that: the reply is cut mid-JSON and the chunk cannot be verified at all.
    Asking for the cap explicitly is what lifts the provider's default, so the
    first call already carries it. ``VERIFIER_MAX_TOKENS`` can lower or raise
    it for an endpoint with a different ceiling.
    """
    # Spelled out rather than read through the constant: the env-guide test finds
    # variables by the literal name at the getenv call, and a name that only
    # exists as a constant would look unread to it.
    raw = os.getenv("VERIFIER_MAX_TOKENS", "").strip()
    if not raw:
        return RAISED_OUTPUT_CAP
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "{}={!r} is not a whole number; using {}",
            _VERIFIER_MAX_TOKENS_ENV,
            raw,
            RAISED_OUTPUT_CAP,
        )
        return RAISED_OUTPUT_CAP
    return value if value > 0 else RAISED_OUTPUT_CAP


def _raised_output_cap(cap: int) -> int:
    """The budget one recovery retry may ask for, given the cap it used first.

    Twice the cap that was just cut, so a verdict that overran the budget has
    room, never below :data:`RAISED_OUTPUT_CAP` and never above
    :data:`VERIFIER_MAX_TOKENS_CEILING`. A cap already at that ceiling is kept
    rather than lowered: the retry is then a second sample, not a bigger one.
    """
    raised = min(max(cap * 2, RAISED_OUTPUT_CAP), VERIFIER_MAX_TOKENS_CEILING)
    return max(raised, cap)


def _target_forms(expected: str) -> list[str]:
    """The accepted target renderings of one glossary pair.

    A planner that writes ``"bảo mật / an toàn thông tin"`` has named two
    acceptable renderings, not one string containing a slash. A translation
    using either is correct, so the whole phrase and each alternative are all
    accepted; the exact phrase stays first because it is the preferred form.
    """
    forms = [expected]
    forms.extend(part.strip() for part in expected.split(" / ") if part.strip())
    return forms


def _normalize_term(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern[str]:
    """Whole-word matcher for a glossary term, tolerant of line wrapping.

    A plain substring test made ``low`` fire on ``below``/``allow``/``flow`` and
    ``date`` on ``update``/``candidate``: the chunk was then told to use a term
    it never contained, which no retranslation can satisfy.
    """
    body = r"\s+".join(re.escape(part) for part in _normalize_term(term).split(" "))
    return re.compile(rf"(?<!\w){body}(?!\w)")


#: Fenced blocks and inline code: a glossary term found only here is an
#: identifier or a sample, not prose the translator has to render.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+|[a-z][A-Z][A-Za-z0-9]*)\b")

#: The words inside an identifier: ``keySet`` is ``key`` + ``set``. Needed
#: because the term and the identifier are compared as words, not as strings —
#: ``"key" in {"keyset"}`` is false, which used to make the ambiguity check
#: silently never fire.
_IDENTIFIER_SPLIT_RE = re.compile(r"[a-z0-9]+(?=[A-Z])|[A-Z]?[a-z0-9]+|[A-Z]+")


def _identifier_words(text: str) -> set[str]:
    return {
        part.casefold()
        for match in _IDENTIFIER_RE.finditer(text or "")
        for part in _IDENTIFIER_SPLIT_RE.findall(match.group(0))
    }


def _prose_only(text: str) -> str:
    """*text* with code fences and inline code removed: what is actually prose."""
    return _INLINE_CODE_RE.sub(" ", _FENCE_RE.sub(" ", text or ""))


def glossary_defects(
    source: str, translation: str, pairs: dict[str, str] | None
) -> list[dict[str, str]]:
    """Deterministic glossary check: a term present in the source must have its
    planned target form in the translation.

    The LLM verifier is good at judging meaning and completeness; it is not a
    reliable glossary lookup. Making the term check here means a missing term
    is reported with the exact pair, so the graph can apply a targeted repair
    instead of re-translating (and possibly losing) the whole chunk.

    Presence in the source is matched on whole words **in prose**: a term that
    only ever appears inside a code fence, an inline code span or an identifier
    (``key`` in ``keySet``, ``index`` in a sample) is code, and leaving it in
    English there is correct. Presence of the target form stays a substring
    test, so an inflected or compounded rendering (``thấp hơn`` for ``thấp``)
    still counts as using the term.

    A term that appears both in prose and inside an identifier somewhere in the
    chunk is ambiguous — the translator may be right to keep the identifier in
    English — so it is reported at ``minor`` severity: visible on the record,
    never enough on its own to re-translate the chunk.
    """
    if not pairs:
        return []
    source_key = _normalize_term(_prose_only(source))
    translation_key = _normalize_term(translation)
    identifiers = _identifier_words(source)
    defects: list[dict[str, str]] = []
    for term, expected in pairs.items():
        term, expected = str(term).strip(), str(expected).strip()
        if not term or not expected or not _term_pattern(term).search(source_key):
            continue
        if any(_normalize_term(form) in translation_key for form in _target_forms(expected)):
            continue
        defects.append(
            {
                "severity": "minor" if _normalize_term(term) in identifiers else "major",
                "type": "terminology",
                "term": term,
                "expected": expected,
                "message": f"glossary term {term!r} must be {expected!r}",
            }
        )
    return defects


_HEAD_FULL_RE = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)
_BULLET_HEAD_RE = re.compile(r"^[•●◦·\-*]\s")
_SENTENCE_END = ".!?:;…"


def _reads_on(prev_text: str, cur_text: str) -> bool:
    """True when *cur_text* reads as the rest of *prev_text*'s sentence: the
    previous line did not end the sentence and this one continues it (a
    lowercase word or a bullet). Mirrors the extractor's own rule."""
    prev = (prev_text or "").rstrip()
    if not prev or prev[-1] in _SENTENCE_END:
        return False
    head = (cur_text or "").lstrip()[:1]
    return bool(head) and head.islower()


def _is_continuation(prev_heading: str, heading: str) -> bool:
    """Whether *heading* is the wrapped tail of *prev_heading* rather than a
    title of its own. A bullet glyph never opens a title, so it always marks a
    continuation; otherwise the continuation must start lowercase after an
    open-ended line. A real title after a real title starts with a capital and
    never merges, so a dropped heading still fails the check."""
    if not prev_heading:
        return False
    if _BULLET_HEAD_RE.match(heading):
        return True
    return _reads_on(prev_heading, heading)


def heading_defects(source: str, translation: str) -> tuple[list[str], list[str]]:
    """Compare heading structure, forgiving the source's own artifacts.

    Returns ``(problems, notes)``: *problems* are hard failures for
    :func:`hard_checks`; *notes* explain a difference that is the extractor's
    pseudo-heading artifact rather than the translator's fault, and are meant
    for the record as source-side warnings.

    The alignment lets one translated heading stand for a *group* of adjacent
    source headings, but every extra heading inside a group must read as the
    wrapped tail of the one before it (a bullet or a lowercase continuation).
    A dropped real heading -- e.g. a translator skipping ``Patent Disclosure
    Notice`` between two titles -- has no such reading and still fails, so the
    re-translation feedback stays honest.
    """
    src = [(len(m.group(1)), m.group(2).strip()) for m in _HEAD_FULL_RE.finditer(source)]
    tgt = [(len(m.group(1)), m.group(2).strip()) for m in _HEAD_FULL_RE.finditer(translation)]
    if [lvl for lvl, _ in src] == [lvl for lvl, _ in tgt]:
        return [], []
    if _aligns_with_joins(src, tgt):
        note = (
            "source heading artifact forgiven: the extractor split an oversized"
            " title/heading line, and the translation re-joined it"
        )
        return [], [note]
    return ["heading structure changed"], []


def _aligns_with_joins(src: list[tuple[int, str]], tgt: list[tuple[int, str]]) -> bool:
    """Whether the translation's heading sequence matches the source's when
    adjacent source headings may be joined into one (small memoized search;
    heading counts are tiny)."""
    n, m = len(src), len(tgt)
    memo: dict[tuple[int, int], bool] = {}

    def solve(i: int, j: int) -> bool:
        if i == n:
            return j == m
        if j == m:
            return False
        key = (i, j)
        if key in memo:
            return memo[key]
        res = False
        if src[i][0] == tgt[j][0]:
            res = solve(i + 1, j + 1)
            k = i + 1
            while not res and k < n and _is_continuation(src[k - 1][1], src[k][1]):
                # src[i..k] becomes one heading in the translation.
                res = solve(k + 1, j + 1)
                k += 1
        memo[key] = res
        return res

    return solve(0, 0)


def heading_notes(source: str, translation: str) -> list[str]:
    """Source-side warnings for a heading difference that was forgiven."""
    return heading_defects(source, translation)[1]


#: What a verdict's reasons sound like when the model is actually *passing* the
#: chunk: it enumerates the checks it cleared. A "fail" whose every reason
#: talks like this — and that names no defect — is a contradiction, not a
#: rejection; there is nothing a re-translation could be told to fix.
_PASS_CLAIM_RE = re.compile(r"\bchecks?\s+(?:are\s+)?pass(?:es|ing|ed)?\b", re.IGNORECASE)


def _contradictory_verdict(reasons: list[str]) -> bool:
    """Whether *reasons* claim the checks passed."""
    return any(_PASS_CLAIM_RE.search(reason) for reason in reasons if reason)


class VerificationError(RuntimeError):
    """Verification could not establish that the translation is safe to ship."""


class TranslationRejected(VerificationError):
    """A completed verification rejected the chunk after its retry budget."""


class VerifyResult:
    """Outcome of verification: pass/fail plus human-readable reasons."""

    def __init__(
        self,
        ok: bool,
        reasons: list[str],
        *,
        skipped: bool = False,
        defects: list[dict[str, str]] | None = None,
        profile: str = "body",
        source_defect: bool = False,
        raw_verdict: str = "",
    ) -> None:
        self.ok = ok
        self.reasons = reasons
        self.skipped = skipped
        self.defects = list(defects or [])
        self.profile = profile
        self.source_defect = source_defect
        #: The verifier reply as it arrived, before any post-processing: the
        # only record of *why* a chunk was rejected that nothing has reshaped.
        self.raw_verdict = raw_verdict
        self.severity = max(
            (str(d.get("severity", "major")) for d in self.defects),
            key=lambda s: _SEVERITY_ORDER.get(s, 1),
            default="none" if ok else "fatal",
        )
        # A source-structure defect cannot be repaired by asking the translator
        # again; neither can a verdict whose only complaints are stylistic. A
        # bare old-style rejection (no defects) still gets the usual retry.
        if ok or skipped or source_defect:
            self.retryable = False
        elif not self.defects:
            self.retryable = True
        else:
            self.retryable = self.severity in {"fatal", "major"}

    def __bool__(self) -> bool:
        return self.ok and not self.skipped

    def __repr__(self) -> str:
        return f"VerifyResult(ok={self.ok}, skipped={self.skipped}, reasons={self.reasons})"


def hard_checks(source: str, translation: str) -> list[str]:
    """Deterministic checks: placeholders and heading levels must survive.

    The heading comparison forgives the source's own pseudo-heading artifacts
    (see :func:`heading_defects`): a faithful translation of a line the
    extractor wrongly split is not the translator's defect, and failing it
    only burns the retry budget on a chunk that can never pass.
    """
    problems: list[str] = []
    if sorted(_PH_RE.findall(source)) != sorted(_PH_RE.findall(translation)):
        problems.append("code placeholders missing or altered")
    problems.extend(heading_defects(source, translation)[0])
    return problems


def _profile_note(profile: str) -> str:
    """Genre-specific instructions for the verifier."""
    if profile == "references":
        return (
            "Profile: references. Article/book titles, journal names, author "
            "names, publisher names and chemical names may stay in the source "
            "language; do not report them as language or terminology defects. "
            "Check completeness, numbering, placeholders and structure.\n\n"
        )
    if profile == "glossary":
        return (
            "Profile: glossary. Source headwords may appear beside their "
            "translation; check that every definition is translated and that "
            "the entry order is intact.\n\n"
        )
    if profile == "code":
        return "Profile: code. Do not check language or terminology; check structure only.\n\n"
    if profile == "table":
        return (
            "Profile: table. Check cells, rows, numbers and placeholders; "
            "style is not a defect.\n\n"
        )
    return ""


def _parse_defects(raw: Any, reasons: list[str]) -> list[dict[str, str]]:
    """Normalise a verifier reply's defects, falling back to its reasons."""
    defects: list[dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "major")).lower()
            if severity not in _SEVERITY_ORDER:
                severity = "major"
            defects.append(
                {
                    "severity": severity,
                    "type": str(item.get("type", "other")),
                    "message": str(item.get("message", "")),
                    "source_quote": str(item.get("source_quote", "")),
                    "translation_quote": str(item.get("translation_quote", "")),
                }
            )
    if defects:
        return defects
    # Older/simpler replies carry only reasons. Treat them as major: they cost
    # one repair, but they are not allowed to delete a chunk by themselves.
    return [
        {"severity": "major", "type": "other", "message": reason}
        for reason in reasons
        if str(reason).strip()
    ]


def _filter_profile_defects(defects: list[dict[str, str]], profile: str) -> list[dict[str, str]]:
    """Drop checks that do not apply to a reference list or code block."""
    if profile in {"references", "code"}:
        return [
            defect for defect in defects if defect.get("type") not in {"language", "terminology"}
        ]
    return defects


class VerifierAgent(Agent):
    """LLM-based final QA plus deterministic hard checks."""

    name = "VERIFIER"

    @staticmethod
    def _note_cut(err: TruncatedError, provider: str, model: str, cap: int) -> None:
        """Say which endpoint cut the reply and how the budget was spent."""
        logger.warning(
            "VERIFIER reply cut at cap {} | provider={} model={} "
            "answer_tokens={} thinking_tokens={}",
            cap,
            err.provider or provider,
            err.model or model,
            err.answer_tokens,
            err.reasoning_tokens,
        )

    def _chat_verdict(
        self,
        messages: list[dict[str, Any]],
        *,
        recovery_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Ask for the JSON verdict, recovering truncated replies.

        The call starts at the configured verifier cap. A reasoning model
        charges its thinking to the same output budget, so a verdict cut at
        4096 would cut the next chunk too; the retry below raises the cap for
        this call only.

        A cut reply whose content is already a complete JSON verdict is used as
        is: the answer finished, the thinking is what overran. Anything else is
        re-asked once at twice the cap (bounded by
        ``VERIFIER_MAX_TOKENS_CEILING``). If that is cut too, the budget is not
        the problem — the model's thinking is — so one last ask goes out with a
        minimal verdict: thinking of the same size leaves room for
        ``{"pass": ..., "reasons": [...]}`` where it did not leave room for a
        full defect list. Only after that does the caller see an outage.
        """
        provider = self.client.config.base_url
        model = self.client.config.model
        cap = _cap_for(provider, model)
        try:
            return self.chat_json(messages, max_tokens=cap, **_reasoning_kwargs())
        except TruncatedError as err:
            self._note_cut(err, provider, model, cap)
            complete = _verdict_from_partial(err)
            if complete is not None:
                return complete
            raised = _raised_output_cap(cap)
            logger.warning(
                "VERIFIER reply was cut at the provider's output-token limit; "
                "re-asking with a raised cap ({})",
                raised,
            )
            try:
                return self.chat_json(messages, max_tokens=raised, **_reasoning_kwargs())
            except TruncatedError as again:
                self._note_cut(again, provider, model, raised)
                complete = _verdict_from_partial(again)
                if complete is not None:
                    return complete
                if recovery_messages is None:
                    raise
                # The answer, not the budget, is what keeps running out: the
                # model spends the cap thinking. Ask for the verdict alone —
                # the same thinking now leaves room for the few tokens it needs.
                logger.warning(
                    "VERIFIER reply cut again at {} (thinking={} answer={}); "
                    "asking for a minimal verdict instead",
                    raised,
                    again.reasoning_tokens,
                    again.answer_tokens,
                )
                try:
                    return self.chat_json(
                        recovery_messages, max_tokens=raised, **_reasoning_kwargs()
                    )
                except TruncatedError as last:
                    self._note_cut(last, provider, model, raised)
                    complete = _verdict_from_partial(last)
                    if complete is not None:
                        return complete
                    raise

    def verify(
        self,
        source: str,
        translation: str,
        src_lang: str,
        tgt_lang: str,
        glossary_md: str = "",
        *,
        hard_issues: list[str] | None = None,
        previous_translation: str | None = None,
        profile: str = "body",
        glossary_pairs: dict[str, str] | None = None,
        source_warnings: list[str] | None = None,
    ) -> VerifyResult:
        """Run hard checks first, then an LLM adequacy verdict.

        The whole chunk is sent: a truncation would leave the tail of a long
        chunk unverified, and the chunker has already bounded the size. When
        *glossary_md* is supplied the verdict also covers terminology, which is
        otherwise only policed document-wide by the consistency pass.

        *hard_issues* lets a caller that already ran
        :func:`hard_checks` on this exact pair of strings pass the result in
        instead of paying for the same scan again; ``None`` means "not
        computed yet" and is the only value that triggers a scan, so an empty
        list is a real "no problems".
        """
        problems = hard_checks(source, translation) if hard_issues is None else hard_issues
        if problems:
            defects = [
                {"severity": "fatal", "type": "structure", "message": problem}
                for problem in problems
            ]
            return VerifyResult(
                ok=False,
                reasons=problems,
                defects=defects,
                profile=profile,
                source_defect=bool(source_warnings),
            )

        glossary = ""
        # A reference list is a different genre: article titles, journal names
        # and chemical names legitimately stay in the source language, and the
        # glossary does not govern them. Treating it as body text is how a
        # bibliography was rejected for "leaving source-language text behind".
        apply_glossary = profile in {"body", "glossary"}
        if (
            apply_glossary
            and glossary_md
            and glossary_md.strip()
            and glossary_md.strip() != "(none)"
        ):
            glossary = f"Glossary (the agreed target form of each term):\n{glossary_md}\n\n"
        profile_note = _profile_note(profile)
        patch_review = ""
        if previous_translation is not None:
            patch_review = (
                f"Before terminology patch:\n{previous_translation}\n\n"
                "The translation below is a terminology-only patch. Compare it with "
                "both the source and the previous translation. Reject changes to "
                "meaning, negation, numbers, entities, requirements or scope; reject "
                "added or omitted content. Allow terminology corrections and only "
                "the grammatical adjustments they require. The source is authoritative.\n\n"
            )
        prompt = (
            "You are a translation QA verifier. Decide whether this chunk is "
            "safe to ship.\n"
            f"Source ({src_lang}):\n{source}\n\n"
            f"{patch_review}"
            f"Translation ({tgt_lang}):\n{translation}\n\n"
            + glossary
            + profile_note
            + "Check, in order:\n"
            "1. Completeness - every sentence of the source is represented.\n"
            "2. No additions - nothing was invented that the source does not "
            "say.\n"
            "3. Language - the translation is in the target language, reads "
            "naturally, and leaves no source-language text behind.\n"
            "4. Structure - headings, lists, placeholders (e.g. ⟦CODE_1⟧), "
            "the emphasis pairs (⟦BOLD⟧...⟦/BOLD⟧, ⟦ITALIC⟧...⟦/ITALIC⟧) "
            "and Markdown table rows are intact.\n"
            "5. Terminology - every glossary term that occurs in the source "
            "appears in the translation in the glossary's own target form. "
            "Skip this check when no glossary is given.\n"
            "Pass only when all the applicable checks hold. Fail only for a "
            "defect a reader would notice, never for a style preference you "
            "would have phrased differently. Name the exact text at fault in "
            "each reason: a failure costs one re-translation of this chunk.\n"
            "Classify every defect as fatal (content/structure cannot ship), "
            "major (a reader would notice wrong meaning, untranslated body "
            "text, or a missing required term), or minor (style/word choice).\n"
            "Reply ONLY with one JSON object, no code fence and no prose:\n"
            '{"pass": true|false, "defects": [{"severity": "fatal|major|minor", '
            '"type": "completeness|addition|language|structure|terminology", '
            '"message": "...", "source_quote": "...", "translation_quote": "..."}], '
            '"reasons": ["..."]}\n'
            'Keep the answer short: omit "defects" entirely when the chunk '
            'passes, use at most two "reasons" of one line each, and quote no '
            "more than 60 characters. The verdict is the answer; do not explain "
            "the checks you ran."
        )
        # The same question, with an answer small enough that a thinking-heavy
        # model can still finish it inside one output budget. Glossary terms are
        # checked deterministically below, so the compact ask leaves them out.
        compact_prompt = (
            "You are a translation QA verifier. Answer immediately; do not analyze.\n"
            f"Source ({src_lang}):\n{source}\n\n"
            f"{patch_review}"
            f"Translation ({tgt_lang}):\n{translation}\n\n"
            + profile_note
            + "Would a reader notice wrong meaning, missing or added content, the "
            "wrong language, broken structure, or a missing required glossary term?\n"
            "Reply with exactly one JSON object and nothing else:\n"
            '{"pass": true|false, "reasons": ["one short reason"]}\n'
            "Use an empty reasons list when it passes."
        )
        t0 = time.monotonic()
        logger.info(
            "VERIFIER starting | src_len={} tgt_len={} model={}",
            len(source),
            len(translation),
            self.config.model,
        )
        try:
            data = self._chat_verdict(
                [{"role": "user", "content": prompt}],
                recovery_messages=[{"role": "user", "content": compact_prompt}],
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "VERIFIER failed after {:.1f}s | error={} model={}", elapsed, exc, self.config.model
            )
            raise VerificationError(f"verification unavailable: {exc}") from exc
        # The raw verdict is the only record of *why* a chunk was rejected:
        # post-processing (profile filters, glossary checks) reshapes the rest.
        logger.info("VERIFIER verdict | {}", json.dumps(data, ensure_ascii=False)[:2000])
        if not isinstance(data, dict) or type(data.get("pass")) is not bool:
            raise VerificationError("invalid verifier verdict: pass must be a boolean")
        reasons = data.get("reasons", [])
        if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
            raise VerificationError("invalid verifier verdict: reasons must be a list of strings")
        raw_defects = data.get("defects")
        named = isinstance(raw_defects, list) and any(
            isinstance(item, dict) for item in raw_defects
        )
        if not data["pass"] and not named and _contradictory_verdict(reasons):
            # The verdict fails while its own reasons claim every check passes
            # and it names no defect: a re-translation cannot fix what the
            # verdict cannot point at, so treating it as a pass keeps the run
            # from burning retries and flagging the chunk for nothing. This
            # must run before the reason fallback below, which would turn the
            # self-praise into an invented "major" defect.
            ok = True
            defects: list[dict[str, str]] = []
            logger.warning(
                "VERIFIER contradiction | pass=false but reasons claim the "
                "checks pass and no defect was named; treating as pass"
            )
        else:
            # Reasons become defects only for a fail; on a pass they are praise.
            defects = _parse_defects(raw_defects, reasons if not data["pass"] else [])
            defects = _filter_profile_defects(defects, profile)
            if apply_glossary:
                term_defects = glossary_defects(source, translation, glossary_pairs)
                if term_defects and data["pass"]:
                    # Without this the run log shows a chunk rejected while the
                    # only verdict on record says it passed.
                    logger.warning(
                        "VERIFIER glossary override | the model passed the chunk but "
                        "{} glossary term(s) are missing: {}",
                        len(term_defects),
                        "; ".join(d["message"] for d in term_defects[:3]),
                    )
                defects.extend(term_defects)
            ok = bool(data["pass"])
            # A "fail" made only of minor notes is a style preference, not a
            # deliverable failure. Keep the translation and report the notes.
            if not ok and defects and all(d.get("severity") == "minor" for d in defects):
                ok = True
            # Every defect the model named is exempt for this genre (a reference
            # list may keep English titles): nothing is left to retranslate.
            if not ok and named and not defects:
                ok = True
            if defects and any(d.get("severity") in {"fatal", "major"} for d in defects):
                ok = False
        logger.info(
            "VERIFIER completed | ok={} elapsed={:.1f}s model={}",
            ok,
            time.monotonic() - t0,
            self.config.model,
        )
        messages = [str(d.get("message", "")).strip() for d in defects]
        messages = [message for message in messages if message]
        if ok:
            all_reasons = list(reasons)
        else:
            # A rejection leads with what is wrong: the graph quotes the first
            # three reasons into the retranslate feedback and the error record,
            # and the model's own summary of the checks it cleared is praise.
            all_reasons = list(messages)
        for message in messages if ok else reasons:
            if message and message not in all_reasons:
                all_reasons.append(message)
        return VerifyResult(
            ok=ok,
            reasons=all_reasons,
            defects=defects,
            profile=profile,
            raw_verdict=json.dumps(data, ensure_ascii=False),
        )
