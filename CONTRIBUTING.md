# Contributing

## Setup

Follow README.md's Setup section for the conda environment and the frontend install; this
document does not restate it.

## Gates a change must pass

The commands below are CLAUDE.md's Commands block, verbatim, the same gate the project itself
runs a change against:

```bash
conda activate tcip-agent          # Python 3.12; torch installs CUDA by default, runs without a GPU
pytest tests/ -n 4 --tb=short --timeout=300 -q
ruff check .
mypy                               # roots from mypy.ini, run from the repo root
python scripts/list_tools.py       # the MCP tool list (never hardcode counts in docs)
python scripts/gate_baseline.py --out <dir>   # the CI-parity gate, Git Bash on Windows
cd packages/tcip-web/frontend && npm run format:check && npm run lint && npm run typecheck && npm test && npm run build
python -m tcip_web                 # backend plus built UI at http://127.0.0.1:8765
python scripts/export_store.py <root>   # a root's database-held records back out as files
python scripts/adopt_store.py <root>    # a root's loose record files into its database
```

`conda activate tcip-agent` is the environment every other line below runs inside, not a gate of
its own. `pytest tests/` is the suite; it binds one storage backend per run (an unset environment
binds the database, `TCIP_STORE_BACKEND=file` the loose-file layout), so a change touching the
storage seam runs it both ways. `ruff check .` and `mypy` are the lint and type gates. `python
scripts/list_tools.py` is how you find the current MCP tool count and names; never hardcode a
count in a doc, comment, or commit message.
`python scripts/gate_baseline.py --out <dir>` runs the same stages `.github/workflows/ci.yml`
declares, so a local pass predicts CI. The frontend line is the frontend's own gate, run only
when a frontend file changed. `python -m tcip_web` is how you confirm a change against the
served app rather than tests alone. `export_store.py` and `adopt_store.py` move a root's records
between the database and loose-file layouts; run them, not a hand-written script, whenever a
change needs to inspect or convert a root's on-disk state.

## Rules a contributor meets

- Tests construct their inputs through the platform's own producers (the MCP tools, the
  pipeline functions), never a hand-built fixture standing in for one.
- A test that guards a fix is shown failing without the fix: `python
  scripts/prove_test_fails_before.py <testfile> -k <expr>`. Its four verdicts are `GUARDS`
  (the recorded failure is the assertion the test names; only this counts as evidence),
  `VACUOUS` (the test passed even without the fix), `INDETERMINATE` (the baseline is not shown
  to precede the change), and `REFUSED` (nothing was selected, or collection failed). Only
  `GUARDS` is reported as a guard; state a `VACUOUS` result as vacuous, not as passing coverage.
- Persisted formats are frozen. `frozen-formats.json`, generated from the store registry by
  `scripts/generate_frozen_manifest.py` and held to it by `tests/test_frozen_manifest.py`, is
  the shipped freeze commitment: every store's classification and version ceiling. A format
  bump is a deliberate, reviewed change that states its own obligations (how the new shape
  coexists with what a reader already wrote), never a silent shape change inside a frozen
  version. There are no runtime migration shims: existing state is conformed by a one-off
  operator script, not by a fallback the running platform carries forward.
- A change touching a persisted field, a refusal, an operating-point stamp, or a delivery gate
  takes design review before code. Open an issue describing the change first; do not send a
  pull request for one of these without a design discussion already open.
- One concern per commit, in dependency order, with LF line endings. A commit message states
  the standing constraint the change installs, not a narrative of the session that wrote it.
- The crop trait vocabulary is controlled: it lives in `crops.yml` and
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`. Do not invent a trait name, state, or
  column prefix anywhere else; thread a real trait through as data read from the project's own
  registry.
- Agent-driven contributions are welcome under the same gates as a human's: the same tests, the
  same review, the same design-first rule for a persisted-format or refusal change. A pull
  request is judged on what it changes, not on who or what wrote it.
