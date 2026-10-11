# Contributing

## Setup

Follow README.md's Setup section for the conda environment and the frontend install; this document
does not restate it. The `tcip` console script (`tcip doctor`, `tcip dump-store`, and every other
operator command) is declared in `packages/tcip-web/pyproject.toml`; an existing editable checkout
gains it only on the next `pip install -e packages/tcip-web`. Every test spawns a command as
`python -m tcip_web.cli <command>` instead, so the test suite holds without that reinstall.

## Gates a change must pass

The commands below are CLAUDE.md's Commands block, verbatim, the same gate the project itself runs a
change against:

```bash
conda activate tcip-agent          # Python 3.13; torch installs CUDA by default, runs without a GPU
pytest tests/ -n 4 --tb=short --timeout=300 -q
ruff check packages tests tools
mypy                               # roots from mypy.ini, run from the repo root
python tools/list_tools.py         # the MCP tool list (never hardcode counts in docs)
npm --prefix packages/tcip-web/frontend run build   # lint, typecheck and test take the same prefix
python -m tcip_web                 # backend plus built UI at http://127.0.0.1:8765
tcip dump-store <project> <out_dir>   # a project's records and logs written out as files
```

`conda activate tcip-agent` is the environment every other line below runs inside, not a gate of its
own. `pytest tests/` is the suite. `ruff check packages tests tools` and `mypy` are the lint and
type gates. `python tools/list_tools.py` is how you find the current MCP tool count and names; never
hardcode a count in a doc, comment, or commit message. `python tools/gate_baseline.py --out <dir>`
runs the same stages `.github/workflows/ci.yml` declares, so a local pass predicts CI. The frontend
line is the frontend's own gate, run only when a frontend file changed; `lint`, `typecheck` and
`test` run under the same prefix. `python -m tcip_web` is how you confirm a change against the
served app rather than tests alone. `tcip dump-store <project> <out_dir>` writes a project's records
and logs out as files; run it, not a hand-written script, whenever a change needs to inspect a
project's stored state.

## Rules a contributor meets

- Tests construct their inputs through the platform's own producers (the MCP tools, the pipeline
  functions), never a hand-built fixture standing in for one.
- A test that guards a fix is shown failing without the fix:
  `python tools/prove_test_fails_before.py <testfile> -k <expr>`. Its four verdicts are `GUARDS`
  (the recorded failure is the assertion the test names; only this counts as evidence), `VACUOUS`
  (the test passed even without the fix), `INDETERMINATE` (the baseline is not shown to precede the
  change), and `REFUSED` (nothing was selected, or collection failed). Only `GUARDS` is reported as
  a guard; state a `VACUOUS` result as vacuous, not as passing coverage.
- A record is the shape its one producer writes and its one reader reads. A re-shaped record changes
  both together, and the records already written are regenerated or discarded with the change; there
  are no runtime migration shims and no defaults for a field a record lacks.
- A change touching a persisted field, a refusal, an operating-point stamp, or a delivery gate takes
  design review before code. Open an issue describing the change first; do not send a pull request
  for one of these without a design discussion already open.
- One concern per commit, in dependency order, with LF line endings. A commit message states the
  standing constraint the change installs, not a narrative of the session that wrote it.
- The crop trait vocabulary is controlled: it lives in `crops.yml` and
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`. Do not invent a trait name, state, or column
  prefix anywhere else; thread a real trait through as data read from the project's own registry.
- Agent-driven contributions are welcome under the same gates as a human's: the same tests, the same
  review, the same design-first rule for a persisted-format or refusal change. A pull request is
  judged on what it changes, not on who or what wrote it.
