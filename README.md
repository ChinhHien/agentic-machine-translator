# polyglot-lc

**Long-context, multi-agent machine translation CLI** — built on an LLM orchestration
graph (LangGraph), a persistent **Reasoning Bank** (ChromaDB), and **SkillOpt**
(textual-gradient optimization of translator prompts).

```
                         ┌──────────────────────────────────────────────┐
                         │                  polyglot-lc                 │
 .md/.txt/.pdf ──Ingest──▶ NFC + code-shield ──▶ Chunker (tiktoken)     │
                         │        │                  │                  │
                         │        ▼                  ▼                  │
                         │  Summary Tree         Glossary (NMT, ...)    │
                         │        │                  │                  │
                         │        ▼                  ▼                  │
                         │  ┌──────────────── LangGraph pipeline ─────┐ │
   Reasoning Bank ◀──────┼─ │ Planner → Translator → Critic           │ │
   (ChromaDB) ──────────▶┼─ │     ▲            │ (score < threshold)  │ │
                         │  │     │            ▼                      │ │
                         │  │     └──── Refiner (max 2 rounds)        │ │
                         │  │                  │                      │ │
                         │  │                  ▼                      │ │
                         │  │             Verifier ──fail──▶ Retrans. │ │
                         │  │                  │ pass                 │ │
                         │  │                  ▼                      │ │
                         │  │        save_trace → next_chunk          │ │
                         │  └──────────────────┼──────────────────────┘ │
                         │                     ▼                        │
                         │              Assemble → SkillOpt             │
                         │   (skills/<src>-<tgt>/<domain>.json)         │
                         └──────────────────────────────────────────────┘
```

## Features

- **Hybrid 5-agent layout**: Planner / Translator / Critic / Refiner / Verifier.
  Translator, Critic and Refiner form the **read-source group** and share one
  caching-capable provider/model (`SOURCE_GROUP_*`), because they all re-read
  the same original chunk. Planner and Verifier are independent and keep their
  own `PLANNER_*` / `VERIFIER_*` settings. Any per-agent `<NAME>_*` value still
  overrides the group — mix OpenAI, DeepSeek, an Anthropic OpenAI-compatible
  proxy, Ollama, vLLM… freely.
- **Provider-native prompt/context caching**: the read-source agents send an
  identical stable prefix (original text + glossary + strategy) first, with
  dynamic content after it, so the provider can cache the shared context.
  Cache-usage metrics from OpenAI (`prompt_tokens_details.cached_tokens`),
  DeepSeek (`prompt_cache_hit_tokens`) and Anthropic-style responses
  (`cache_read_input_tokens`) are reported separately. `PROMPT_CACHE=explicit`
  additionally emits `cache_control` breakpoints for gateways that need them.
- **PDF via MarkItDown**: PDFs are converted to Markdown with
  [MarkItDown](https://github.com/microsoft/markitdown), then parsed by the
  same structural parser as `.md` files (code shielding included). A legacy
  pypdf→pdfplumber extractor is kept as a fallback.
- **Long-context segmentation**: token-aware chunking (tiktoken, budget from
  `TRANSLATOR_MODEL`) at paragraph/heading boundaries with overlap.
- **Summary tree** (doc → section → chunk) built with the Planner LLM.
- **Glossary extraction**: acronyms, capitalized phrases, terms repeated ≥ 2×,
  plus optional user CSV.
- **Reasoning Bank**: ChromaDB persistent traces `{strategy, few_shot, pitfalls,
  score, ts…}`; upsert gated at score ≥ 0.75.
- **SkillOpt**: per-chunk (prompt, output, critic_score) collection; the Planner
  proposes textual gradients; a skill is promoted only when the mean score of the
  last 20 chunks beats the current skill.
- **Simple, non-dynamic CLI**: one line per running step (`[3] Running critic
  (chunk-000) ...`), then a compact token/request report (input, cached input,
  uncached input, output, total, requests — plus a per provider/model
  breakdown). No spinner, progress bar, live dashboard or redraw. Structured
  events still stream to `logs/run_<ts>.jsonl`.
- **Bounded concurrency**: independent chunks run in parallel up to
  `MAX_CONCURRENCY` (default 2), each keeping its internal
  Translate→Critic→Refiner→Verify order. Output is assembled in the original
  chunk order. Planner / summary / assemble stay sequential.
- **Controlled retries**: infrastructure failures (429/408/timeouts/5xx/
  connection) retry with exponential backoff + bounded jitter
  (`LLM_RETRY_JITTER`) and honour `Retry-After`; semantic failures (Verifier
  FAIL) use a separate, bounded retranslate path.
- **Adaptive refiner**: `REFINE_MIN_SCORE` skips the Refiner for drafts that
  already score high enough (defaults to `CRITIC_THRESHOLD`, so behaviour is
  unchanged unless raised).
- **Cache observability**: per-request `HIT` / `MISS` / `UNKNOWN` derived only
  from provider metadata (absence is never counted as a miss), plus cached /
  uncached input token totals and a light debug fingerprint of the stable
  prefix.
- **Quality gates**: deterministic hard checks (code placeholders `⟦CODE_n⟧`,
  heading levels) + LLM verdict; verifier failure triggers one re-translation
  with the critique attached. Structural failures short-circuit *before* the
  LLM verifier call.

## Install

Requires Python 3.11+.

```bash
pip install -e ".[dev]"        # or: uv pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` → `.env` and fill in per-agent settings:

```dotenv
PLANNER_BASE_URL=https://api.openai.com/v1
PLANNER_MODEL=gpt-4o-mini
PLANNER_API_KEY=sk-...
PLANNER_TEMPERATURE=0.2

TRANSLATOR_BASE_URL=https://api.openai.com/v1
TRANSLATOR_MODEL=gpt-4o
TRANSLATOR_API_KEY=sk-...
TRANSLATOR_TEMPERATURE=0.3

CRITIC_BASE_URL=https://api.deepseek.com/v1
CRITIC_MODEL=deepseek-chat
CRITIC_API_KEY=sk-...
CRITIC_TEMPERATURE=0.1

# ... REFINER_*, VERIFIER_* (see .env.example), plus:
EMBED_BASE_URL=https://api.openai.com/v1
EMBED_MODEL=text-embedding-3-small
EMBED_API_KEY=sk-...
```

Missing required keys raise a clear error shown on the red **Errors panel** —
nothing is hard-coded; everything comes from env.

### Keyless demo / offline mode

Set `LLM_MOCK=1` to run the whole pipeline against deterministic in-process
mocks (no API keys, no network). Embeddings fall back to deterministic
hash vectors when the embedding endpoint is unreachable.

## Usage

python -m venv .venv
.venv\Scripts\Activate.ps1

```bash
polyglot translate --file sample.md --src en --tgt vi
polyglot translate --file report.pdf --src auto --tgt en --domain legal \
                   --glossary glossary.csv --out report.en.md --max-tokens 512
```

| Option | Description |
| --- | --- |
| `--file / -f` | Input `.md`, `.txt` or `.pdf` |
| `--src / -s` | Source language, `auto` = langdetect |
| `--tgt / -t` | Target language code |
| `--domain / -d` | Domain hint (tech, legal…) used by Planner/RB/SkillOpt |
| `--glossary / -g` | CSV glossary `src,tgt` per row |
| `--out / -o` | Output path (default `<stem>.<tgt><ext>`) |
| `--max-tokens / -m` | Chunk token budget override |

The final report prints: chunk count, average critic score, retries, traces
stored, skill promotion, output file, JSONL log path and the token/request
summary.

## Demo without keys

```bash
LLM_MOCK=1 polyglot translate --file sample.md --src en --tgt vi
```

You will see one line per running step followed by the report:

```
  [1] Running ingest ...
  [2] Running segment ...
  ...
Completed.

Token usage:
  Input:  11,963
  Output: 248
  Total:  12,211
  Requests: 15
```

## Tests

```bash
pytest -q
```

Covers ingestion (md/txt/registry, code shielding, NFC), chunker budgets &
overlap, glossary, config validation, LLM client against a mocked HTTP API
(`respx`), Reasoning Bank retrieve/upsert gating + persistence, SkillOpt
promotion logic, TM, router rules and an end-to-end mock pipeline.

## Project layout

```
src/polyglot_lc/
├── cli.py                 # typer CLI
├── config.py              # per-agent + source-group env config, cache policy
├── pipeline.py            # end-to-end orchestration entry
├── ingest/                # registry, pdf (MarkItDown→markdown), txt, md
├── segment/               # chunker, summary_tree, glossary
├── agents/                # base, context, planner, translator, critic, refiner, verifier
├── orchestrator/          # LangGraph state, router, graph
├── reasoning_bank/        # chromadb store, embedder, trace schema
├── skillopt/              # optimizer, skill_store
├── memory/tm.py           # translation memory
├── llm/client.py          # OpenAI-compatible client (+ cache metrics, retry, mock)
└── ui/                    # simple step reporter, step context, JSONL events
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the pipeline diagram,
step-by-step processing flow, and why each agent/tool is used.
