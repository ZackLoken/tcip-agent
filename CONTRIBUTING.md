# Contributing

## Setup

Follow README.md's Setup section for the conda environment and the frontend install; this document
does not restate it. The `tcip` console script (`tcip dump-store`, `tcip diagnose-project` and
every other command) is declared in `packages/tcip-web/pyproject.toml`; an existing editable
checkout gains it only on the next `pip install -e packages/tcip-web`. Every test spawns a command
as `python -m tcip_web.cli <command>` instead, so the test suite holds without that reinstall.

## Gates a change must pass

The gate is CLAUDE.md's Commands block, the same one the project itself runs a change against;
this document does not restate it. `python tools/gate_baseline.py --out <dir>` is the local run of
CI's jobs, whose module docstring says which steps it leaves to CI.

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
- A pull request states its line delta by area (`python tools/line_delta.py <repo> main..`) as a
  signal, and leaves the mechanism it touched with less to maintain than it found: no second
  implementation of a fact, nothing written by hand that a library the environment carries
  provides, no vocabulary or key declared in more than one place, no per-shape copy of a reader.
  Growth a pull request justifies is admitted; machinery kept beside its replacement is not.
- Two readers read every pull request before it merges, a code review against the issue's
  done-criterion and the invariants, and a test-engineering read of every guard's fail-before
  verdict and of whether each test reaches the boundary a breeder or the agent touches; their
  verdicts are recorded on the pull request.
- The crop trait vocabulary is controlled: it lives in `crops.yml` and
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`. Do not invent a trait name, state, or column
  prefix anywhere else; thread a real trait through as data read from the project's own registry.
- The platform has one maintainer, and every change lands through the maintainer's own review.
  A pull request from outside is read as a report of what its author needs: it is reviewed,
  what it asks for is re-implemented under these rules when the roadmap takes it, and the pull
  request itself is closed with that answer, never merged. An issue is the better form for the
  same request.
