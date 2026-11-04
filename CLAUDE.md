# CLAUDE.md

The developer's contract for Claude (and any Claude-driven agent) changing this repository. When
this file or the operating contract and a knowledge document disagree on a domain fact, the
document wins; on behavior, this file and the contract win. Machine and harness facts live in
`CLAUDE.local.md` (not shipped); commit, push and prose rules that hold across projects live in the
global `CLAUDE.md`; the rules every change meets live here, and `CONTRIBUTING.md` is the page for
someone outside reporting a problem or asking for a feature.

## The operating contract

The platform's agent works under one operating contract, a knowledge document every terminal
launch renders into the workspace in the form its harness reads. It binds work in this repository
too, and this file restates none of it:

@packages/tcip-mcp/src/tcip_mcp/knowledge/operating-contract.md

## What is settled

Beyond what the operating contract settles, every artifact may be changed or replaced. Do not cite
one as a reason not to change it.

## What this is

`README.md` has the pitch and the roadmap. Four packages under `packages/` share one `.tcip/`
state directory, each with its own `CLAUDE.md` for layout. Domain knowledge lives in
`packages/tcip-mcp/src/tcip_mcp/knowledge/` as repo files and reaches every harness as a skill
naming the `serve_domain_knowledge` tool, which serves it; a document is read in full by every route.

## Working on the platform

- Platform work skips the operating contract's steps that act on a breeder's project:
  `inspect_project`, `load_project_memory`, `report_friction` and `write_retrospective`.
- Orientation reads go by a code search (the semantic index where it is connected, otherwise
  `git grep` or the Grep tool over a named subtree) and then the file itself, whole.

## Invariants for platform code

The operating contract's invariants and these state guarantees the platform owes. They do not
protect the mechanisms that happen to provide those guarantees today: where a tool, file, refusal
or schema currently serves one, that implementation is replaceable and the guarantee is not.

- The platform's own code is the measurement gate, the trait entry, the records and the rails, and
  as little else as possible.
- A required value is a pydantic field or a parameter with no default, and its type is what refuses
  it unstated; nothing is written by hand to say the same. The few hand-written refusals that
  remain each ship with a test, through the platform's own producer, that a legitimate call still
  succeeds.
- Nothing on disk is a consumer. Every project, record, lock, hash, log row and checkpoint that
  exists today is sample data made to exercise the implementation, and a change that re-keys,
  re-shapes or invalidates it conforms or discards it as part of the change. A mechanism is never
  kept, shaped or recommended to preserve existing data, keep old records readable, or migrate them:
  no "predates the field" branch, no default for a record written before a key existed, no hash
  restored so existing locks stay bound, no reader kept for rows already written. A live consumer is
  code with a caller. A sentence in a brief, a report, a recommendation or a review that names
  existing data as a reason to keep something is a defect in that sentence.

## Working a change

- A test that guards a fix is observed failing without the fix, through
  `tools/prove_test_fails_before.py`, and the verdict is named: `GUARDS` is the only evidence, and
  `VACUOUS`, `INDETERMINATE` and `REFUSED` are reported as what they are, never as coverage. Every
  test constructs its input through the platform's own producer (an MCP tool, a pipeline function,
  a route, the document writer), never a hand-built fixture standing in for one.
- A shared fact has one implementation (the operating contract's "call one from the other") and
  ships an agreement test: two routes' own records compared against each other, not each
  against a fixture. The test guards that the routes still agree, and never licenses a second
  implementation to stay. A metric, a derivation or a training mechanism ships the same test
  pointed outward: its number compared against an external reference (pycocotools for average
  precision, scikit-learn for the agreement and classification metrics, a tiny dataset whose right
  answer is known in advance for a training or freezing mechanism), since a suite that compares the
  platform only to itself proves nothing about the number. A change replacing a hand-written
  implementation with a library call lands with that test as its gate.
- Gates before a pull request: the Commands block's lines for what the change touched (the ruff
  and mypy lines for Python, the markdown line for a markdown file, the five frontend lines for a
  frontend file), and the change's own test files, each run to its end and its output read before
  the pull request opens. CI is the full gate. A test touching the filesystem outside `tmp_path` is
  the first reread on a break that only shows in CI.
- A green suite is evidence only about the defects its fixtures can distinguish. Before trusting it
  as proof a change was safe, check that some fixture could have come out differently.
- A guard test reads what the agent or the breeder receives through its producer (the hook's
  orientation, a rendered instruction file, the settings, the argv, a route's answer); a behavior
  no test can reach is named on the pull request with the one check that reaches it.
- Every change leaves the mechanism it touched with less to maintain than it found. The line delta
  by area (`python tools/line_delta.py <repo> main..`) is reported on every pull request as the
  signal that prompts the code reviewer's look, never the test. The merge stops, on any issue, when
  the code reviewer names in the touched mechanism a second implementation of a fact or machinery
  kept beside its replacement: a vocabulary or key declared twice, a per-shape copy of a reader.
- Before changing how a shared fact is spelled, produced or read, enumerate its producers and
  readers in the touched mechanism and change them together; a convention changed at one site and
  not its siblings leaves the fact spelled two ways.
- A read by another model family is advisory on a design and binding on a change to a persisted
  format: when two families agree against the author's position, the author conforms or brings
  the split to the owner, and the outvoted side never lands on its own authority.
- Commits: one concern each, in dependency order, each message stating the standing constraint the
  change installs, never a narrative of the session that made it.
- One fact has one spelling. Whenever a change touches a fact, ask what else states it: a second
  key, field, flag or parameter that must agree with the first is a derived spelling, whether or not
  this change introduced it, and it is deleted rather than kept, reconciled or asked which one
  governs. Delete it at the moment the surface changes, and conform every test config, sample
  project and knowledge sentence that carried it; those are sample data the work rewrites as it
  goes, never an interface whose current shape constrains the change. The commonest case is a record
  that can now carry a fact per item: the parameter that carried it once for the whole run goes,
  since its single value is usually the limit itself, and moving that value onto the new record
  rebuilds the limit one level down. A fact that cannot be re-derived and has nowhere yet to live is
  scheduled work, never a spelling left in place.

## Commands

```bash
conda activate tcip-agent
pytest <the change's test files> --tb=short --timeout=300 -q   # CI runs the whole suite
ruff check packages tests tools
ruff format --check packages tests tools
mypy                               # roots from mypy.ini, run from the repo root
python tools/list_tools.py         # the MCP tool list (never hardcode counts in docs)
npx --yes markdownlint-cli2@0.23.3   # the markdown gate, run from the repo root
npm --prefix packages/tcip-web/frontend run format:check   # the frontend gate, whole and in order
npm --prefix packages/tcip-web/frontend run lint
npm --prefix packages/tcip-web/frontend run typecheck
npm --prefix packages/tcip-web/frontend test
npm --prefix packages/tcip-web/frontend run build
python -m tcip_web                 # backend plus built UI at http://127.0.0.1:8765
tcip dump-store <project> <out_dir>   # a project's records and logs written out as files
```

Every process binds the store once at its entry point (`tcip_store.bind()`); a root's records and
logs live in `<root>/.tcip/store.db`. The MCP server auto-launches from `.mcp.json`; a stale tool
index means restart the client. A project is the only root: the web backend holds the project it has
open, the MCP server takes `--project <path>` at start or, started without one, acts on the first
project its `initialize_project` creates, and every other entry point obtains its project once and
passes it. The repository's own `.mcp.json` names no project, so there the project tools refuse
naming `--project` until a project is created, and the knowledge tools work.

## Conventions

- Lazy-import torch and torchvision inside function bodies, so MCP startup stays fast.
- A standing capability the agent uses is an MCP tool, which every harness lists with its schema
  and runs outside the harness's own sandbox; when an operator runs it too, its console command
  calls the same function. A capability only an operator uses is a console command alone.
- Crop traits are controlled vocabulary in
  `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/crops.yml`; verify there before asserting.
- Every piece of shipped prose (comments, docstrings, log and UI strings, test names, file names,
  scripts, README, skills, package `CLAUDE.md`s) is for whoever reads it next, never a changelog of
  the session that wrote it: no tracking labels and no inline decision dates. If nothing survives
  once that framing is stripped, write nothing.
- A docstring states what the function does and its contract: its inputs, what it returns, what it
  refuses. It never restates a rule from this file or a package `CLAUDE.md`, never names its
  callers, and never explains a rejected alternative; those live in the change's design issue or
  nowhere.
- `docs/` and `.claude/` are local, gitignored dev tooling, except the generated skills under
  `.claude/skills/`, which are tracked.
