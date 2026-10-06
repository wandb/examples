# Guide template

Guides use a terse, command-first register. Structure, in order:

1. **Noun-phrase title** — `# Setup`, `# Tracing`. The folder carries the number.
2. **Opening sentences** — two to four: the problem, the one new idea, honest labels where they apply (Requires a live model / Optional / Advanced / Experimental).
3. **`## How it works`** — two to four sentences of prose, then one or two focused excerpts copied verbatim from the example files. Use a second excerpt only when the idea has two inseparable API moments, such as tool and LLM spans; keep each excerpt near 8–20 lines.
4. **`## Run it`** — the command, exactly as written from the language directory.
5. **`## What you should see`** — the trimmed terminal output and one or two sentences of interpretation: what varies between runs, what doesn't.
6. **`## Inspect it in Weave`** — required in every guide: where to go and two to four bullets of concrete, observable success states. Link the sanitized capture under `assets/expected-output/`; if changed nondeterministic output has not been re-run live, label it as an expected shape instead of a capture.
7. **`## Useful commands`** — optional; only when the guide has real variations (a flag that changes control flow, a config override). Never boilerplate.

What guides do not contain:

- No `**Outcome:**` / `**Requires:**` metadata blocks — fold both into the opening sentences.
- No "Next" footers or per-guide troubleshooting sections — the guide index orders the path and links troubleshooting once.
- No full terminal transcripts — trim to the signal lines; full captures live in `assets/expected-output/`.
- No full example files inline — only the focused API moments. Readers run the example; they don't copy the guide.

Style rules:

- Commands run exactly as written from the language directory (`python/` or `typescript/`).
- Excerpts are verbatim, contiguous lines from the linked example — no `...` placeholders, no secrets, no edits. `scripts/validate_docs.py` enforces this.
- Direct language ("Run the example"), no hype, no fake surprises, no unsupported production claims.
- Link official W&B docs for exhaustive options instead of duplicating them.

The machine-level agent integration in guide 07 is the deliberate structure exception: it must keep the opening boundary warning and **Inspect it in Weave**, but it does not pretend a global plugin install is a repository-run example.
