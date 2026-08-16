# Project guide

This project is now in the paper-writing phase; the core experimentation is
complete.

## Task-specific sources

Do not preload these files. Read only the files relevant to the current task:

- For manuscript edits, read `paper/neurips_2026.tex`. It is the active
  manuscript and the primary source of truth for current paper work.
- For questions about the paper's argument, scope, or non-claims, read
  `paper/paper_plan.md`.
- For experimental conclusions, headline numbers, or claim boundaries, read
  `EXPERIMENT_REPORT.md`.
- For pilot history, failed controls, full result matrices, secondary analyses,
  or detailed audit reasoning, consult `EXPERIMENT_LEDGER.md` only when the task
  requires that detail.
- For implementation and artifact locations, consult `ARTIFACT_INDEX.md`.
  If something still feels missing, feel free to look through the repository.

`VNAA-Proposal.pdf` is historical background only. Do not use it to infer what
was run, what results were obtained, or the paper's current claims.

## Working conventions

- Keep the paper's claims aligned with `EXPERIMENT_REPORT.md` and its explicit
  limitations. Use `EXPERIMENT_LEDGER.md` to resolve detailed audit questions,
  not to broaden the concise report's claim boundaries.
- Preserve double-blind review: do not add names, usernames, repository
  handles, or other identifying information to repository files.
- Do not add `Co-Authored-By: Claude` trailers or other AI self-attribution to
  commits.
- `data/` and `results/` are gitignored. Treat them as local artifacts; do not
  delete or regenerate them unless the task specifically calls for it.
- SAD prompt text must never be written to committed files. See
  `EXPERIMENT_REPORT.md` if work touches Slice 1 data.

## Practical notes

- Use `.venv\Scripts\python`, not the system Python.
- Run `paper/build.ps1` to build the manuscript on Windows.
- Run `.venv\Scripts\python -m pytest` for the project test suite when code
  changes warrant it.
- After completing and verifying requested file changes, automatically commit
  the task-related files and push to GitHub main. Preserve unrelated user changes.
