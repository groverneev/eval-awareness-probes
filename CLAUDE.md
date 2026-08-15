# Project guide

This project is now in the paper-writing phase; the core experimentation is
complete.

## Read first

1. `paper/neurips_2026.tex` is the active manuscript and the primary source of
   truth for current paper work.
2. `paper/paper_plan.md` states the paper's argument, scope, and non-claims.
3. `EXPERIMENT_REPORT.md` is the complete, artifact-grounded record of the
   experiments, results, and limitations. Consult it for experimental or
   numerical questions. If something feels missing, feel free to look through
   the repository.

`VNAA-Proposal.pdf` is historical background only. Do not use it to infer what
was run, what results were obtained, or the paper's current claims.

## Working conventions

- Keep the paper's claims aligned with `EXPERIMENT_REPORT.md` and its explicit
  limitations.
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
