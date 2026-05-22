# AGENTS.md — pyGOTM Agent Operating Contract

This file is mandatory for both Claude Code and Codex. It is intentionally short.
The detailed project controls live in `.superpowers/`. Do not make code changes,
run Python commands, run tests, run validation, run benchmarks, or edit project
files until the read gate below has been satisfied.

## Mandatory Read Gate for Claude Code and Codex

Before any code change or Python-related command, the agent must read:

1. `AGENTS.md` — this operating contract.
2. `.superpowers/agent-read-gates.md` — the task-to-reference-file routing rules.
3. Every task-specific reference file required by `.superpowers/agent-read-gates.md`.
4. The existing project files that will be changed.
5. For any physics, numerical, or translation work, the corresponding Fortran
   source under `gotm-model/code/src/` before implementing or modifying logic.

If the task changes code, tests, validation, benchmarks, configuration, packaging,
or documentation, the final response must include a short "Files reviewed" note
listing the control files that were read. If any required control file was not
reviewed, stop and review it before proceeding.

## Non-Negotiable Project Rules

- Taichi is not used. No `import taichi` is permitted in `src/` or `tests/`.
- Use Numba for computational acceleration.
- Use `np.float64` throughout scientific kernels unless a documented and tested
  exception is approved.
- Scientific parity with Fortran GOTM is required before performance tradeoffs.
- Preserve the Fortran folder/file mapping exactly under `src/pygotm/`.
- Preserve Fortran comments in Python module/function docstrings.
- No hidden simulation state. Results must be reproducible from YAML config.
- The kernel is the product. Keep `src/pygotm/` free of web/UI concerns.
- API and UI are wrappers around the pyGOTM package.
- No unsupported fallback to legacy Python timestep logic during parity runs.

## Conda Environment Rule

All Python commands, test runs, validations, benchmarks, and Numba recompilations
must be prefixed with `conda run -n pygotm` so they execute inside the project
conda environment named `pygotm`. Do not activate or deactivate environments in
the shell.

```bash
conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .
conda run -n pygotm python -m pytest -W error::RuntimeWarning
conda run -n pygotm ruff format .
conda run -n pygotm mypy src/
conda run -n pygotm python script.py
```

Do not rely on any active shell environment. Do not use `.venv`. Do not use `uv`.
Do not use `pip` for dependency management. The only permitted `pip` command is
the no-dependency editable install above, which registers the local checkout so
`import pygotm` and the `pygotm` console script work from a fresh conda
environment.

Canonical environment commands:

```bash
conda env create -f pygotm-conda-env.yml
conda env update -f pygotm-conda-env.yml --prune
conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .
```

## Shell Command Policy

Claude Code and Codex may execute shell commands directly. Keep command usage
targeted and keep output in the main conversation concise. The goal is to minimize irrelevant output, reduce token usage, and use fast, precise tools.

Use `rg` for text search.

- Use `rg -n "pattern"` for source, docs, tests, configs, logs, and scripts.
- Use globs to narrow scope, for example `rg -n "pattern" -g "*.py"`.
- Use limited context only when needed, for example `rg -n -C 2 "pattern"`.

Use `fdfind` for file and directory discovery.

- Use `fdfind <name>` to locate files.
- Use `fdfind -e py` to find Python files.
- Use `fdfind -t d` to find directories.
- Prefer targeted searches over broad recursive listings.

Use `jq` for JSON.

- Use `jq` for `package.json`, lock files, JSON config, logs, and command output.
- Do not parse JSON with `grep`, `sed`, `awk`, or ad hoc scripts when `jq` is practical.

Use `tree` for compact project layout.

- Use `tree -L 2` or `tree -L 3`.
- Avoid deep unrestricted `tree` output.
- Exclude noisy directories when needed.

Use `git` for repository state and changes.

- Use `git status --short`.
- Use `git diff --stat` before full diffs.
- Use `git diff -- <path>` for targeted diffs.
- Use `git ls-files` when only tracked files matter.

Use `batcat --paging=never -n` for readable file excerpts with line numbers.

Avoid broad directory dumps, full
logs, full test output, full diffs, long benchmark tables, repeated warning
blocks, and irrelevant stdout/stderr noise unless the user explicitly asks for
full output.

When reporting command results, include only the details needed for the next
action: command intent, exit status, concise stdout/stderr summary, exact error
excerpts, and files changed when applicable.

## Destructive Command Safety

Do not run destructive commands unless explicitly approved by the user.

Do not run without explicit approval:

- `rm -rf`
- `git push`
- `git commit`
- `git reset --hard`
- `git clean -fd`
- force pushes
- credential/token printing commands
- network download or install commands

For risky commands, summarize the intended command and ask for approval first.

## Standard Quality Gate

Before declaring code work complete, run the applicable gate from
`.superpowers/development-workflow.md`. The default gate for code changes is:

```bash
conda run -n pygotm python -m pytest -W error::RuntimeWarning
conda run -n pygotm mypy src/
conda run -n pygotm ruff format .
conda run -n pygotm ruff check .
```

Use project-specific variants when the task is narrower, but never skip relevant
validation silently.

## Detailed Reference Files

Detailed project requirements are split into:

- `.superpowers/agent-read-gates.md`
- `.superpowers/pygotm-architecture.md`
- `.superpowers/gotm-translation-map.md`
- `.superpowers/translation-rules.md`
- `.superpowers/implementation-plan.md`
- `.superpowers/testing-and-validation.md`
- `.superpowers/known-gotm-sign-conventions.md`
- `.superpowers/development-workflow.md`
