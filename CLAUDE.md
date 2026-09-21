# CLAUDE.md

Operating contract for Claude (and any Claude-driven agent) working in this repo: behavior and
invariants, not documentation. When this file and a skill disagree on a domain fact, the skill
wins; on behavior, this file wins. Machine and harness facts live in `CLAUDE.local.md` (not
shipped); commit, push and prose rules that hold across projects live in the global `CLAUDE.md`.

While the architecture contraction is underway, `docs/NEXT_SESSION_PROMPT.md` governs procedure
only: which session does what, in what order, with which tools. This file governs what a change
is and how it is judged, and the prompt never overrides it; a prompt sentence that would change
how a change is judged is a conflict brought to the owner rather than followed.

## The foundation

The platform's job is to give the agent the facts it cannot otherwise know (what primitives exist
and their interfaces, the trait semantics, the scientific rails, the objective, the data in hand)
and then rely on the agent's own CV-scientist reasoning for a problem no one wrote a procedure
for. It must not hand the agent recipes, prescribed pipelines, or "for trait X do Y". The test for
every skill, tool, doc and code path: does it leave room to reason from facts, rails and a
discoverable toolkit, or does it box the agent into a method? Trait semantics stay defined.

Only these are settled: `crops.yml` is the trait authority; PyTorch, TensorBoard and Ray Tune are
the technology choices. Every other artifact may be changed or replaced. Do not cite one as a
reason not to change it.

## What this is

TCIP is an agentic ML/CV platform for automated phenotyping in tree-crop breeding, with you as the
ML/CV engineer driving it. Scope today is 2D imagery, RGB and N-channel; 3D point clouds are not
built. `README.md` has the pitch and the roadmap. Four packages under `packages/` share one
`.tcip/` state directory, each with its own `CLAUDE.md` for layout. Domain knowledge lives in
`packages/tcip-mcp/src/tcip_mcp/knowledge/` as repo files and reaches every harness through
generated skills or the `serve_domain_knowledge` tool; a document is read in full by every route.

## Operating posture

Your default failure is pushing through friction by guessing.

- Project work confirms which project is active before touching data, and reads that project's own
  memory first. Platform work skips the project ritual entirely.
- Report friction through `report_friction` the moment you hit it: a missing tool, ambiguous data,
  an op that failed twice, a decision needing human judgment, behavior that surprised you. The
  free-text detail matters more than the category. A mandated action that is blocked is itself a
  report, never a silent skip. End substantial work with `write_retrospective`.
- Never state a fact about this codebase, a domain, or a workflow that you have not executed or
  read this session. One docstring, one sample project, one capture rig describes that instance,
  not the platform's general case; ask before generalizing. A claim about purpose is checked by
  testing its premise.
- Semantic search first, grep second. When a claude-context server is configured, its search is the
  default for any exploratory or orientation read: it finds concept-shaped matches grep cannot, and
  it is much faster than sweeping the tree. Reach for `git grep` once you know the exact string you
  are pinning, which is a symbol, a call site or a line number. Do not default to grep for
  everything. The index reflects the last rebuild, so uncommitted code is grep-only, and `docs/` is
  not indexed at all.
- Progressive disclosure: start simple; add complexity only when data or metrics justify it.

## Invariants that protect the science

These state guarantees the platform owes. They do not protect the mechanisms that happen to
provide those guarantees today: where a tool, file, refusal or schema currently serves one, that
implementation is replaceable and the guarantee is not.

- Measurement integrity is the highest rule. The domain expert defines each trait's measurement;
  you operationalize their definition and never substitute your own, and when it is unclear you
  stop and ask. A delivered number requires a breeder-confirmed operationalization and a
  validation against a reference sized to the trait, with the provenance recording which reference
  answered for it. No validated measurement, no result. Geometry measures dimensions on a
  validated mask with scale calibration; it never stands in for finding the object or judging a
  biological state. Tentative domain logic, whatever made it tentative, is labelled tentative and
  validated or removed; it never becomes institutional truth by reuse.
- Scientific defensibility: every phenotype reproducible and auditable end to end, from data and
  environment through predictions and operating point to the measurement. Parameters are derived
  from the data at runtime, never frozen constants; when a threshold varies by dataset, model or
  trait, the deliverable is the capability to derive it, never the value.
- Agent-legible and breeder-coherent: a discoverable toolkit with docs that match the code, and a
  GUI that guides the breeder without stranding them, at equal weight.
- Every state change leaves a record, and a record that cannot be written is raised rather than
  swallowed. Experiments are immutable: a new run, never an overwritten one.
- A subject is an object class to isolate, not a trait.
- No pilot vocabulary as framing: a trait's own name, state or column prefix never names a general
  mechanism in identifiers, comments or docs; thread the real trait through as data from the
  project's registry. A concrete trait is fine as one marked example.
- A negative is an empty label file plus a human marking the image done with nothing on it.
  An empty label file alone is never a negative, and empty label files are never deleted without
  asking.
- Never train or evaluate on a format the data does not positively confirm; refuse rather than
  guess.
- Confirm before destructive or outward actions (deleting labels, overwriting weights, exporting
  deliverables); approval for one does not extend to the next.
- Enumerate the consumers before deleting anything; a deleted assertion's fact needs a new home.
- When two code paths must agree, call one from the other. A consistency check whose two sides
  share an implementation proves nothing.
- A rail must admit valid work, not only reject invalid work: every refusal ships with a test
  proving a legitimate call still succeeds, constructed through the platform's own producer.
- No silent fallback when required information is missing: require it or refuse, naming the real
  primitive. A guessed value that can reach a delivered result is a fabrication with a warning
  attached. A refusal is the last resort, never the design: the goal is fewer places where a
  fact can go missing, so a fact read in several places gets one producer and one path its
  readers share, and one check at that boundary, rather than a refusal inside each reader.
  Several readers each asking the same question is the defect, and adding a refusal to each is
  how it survives.
- A stated format, subject or root is a claim the data must positively carry, never one it merely
  fails to contradict.

## Pipelines, models, seeing

No universal pipeline: derive the decomposition from the data in hand (`pipeline-design` skill);
`toolkit-inventory` maps the pieces that already exist so you do not rebuild one. You can see
images. The platform's renderers write to `.tcip/artifacts/viz/`; read the path with your
image-capable tool, describe what you see, then recommend. External phenotyping resources are read
for general techniques only, never for a per-trait pipeline; the endpoint is a trained model.

## Working a change

- A contraction change starts from its deletion list: the functions, keys, branches and records
  it removes, named with their line counts read from the tree, and what replaces each. The
  change is done when each is gone and nothing was added beside the replacement. The line delta by
  area (`packages/`, `tests/`, other, off `git diff --numstat`, run by `tools/line_delta.py`) is
  evidence that the list was executed, never the target: package code that grew means a mechanism
  survived beside its replacement and is named and deleted or the change stops, and a delta that
  fell without a named deletion is the same finding the other way. A finding names a site; the fix
  names the mechanism that produced the site and removes it. Patching the site with a refusal, a
  second comparison, a lookup by another spelling or a per-shape copy of an existing reader keeps
  the mechanism and is how it survives to the next read. Before proposing any fix, say what
  deleting the mechanism would look like and what stops that; only a live consumer or an owner
  ruling stops it, never the code's current shape. A brief never scopes out a sibling of a
  mechanism the change deletes: if every instance cannot go, that is the stop rule, brought to the
  owner, never a paragraph naming what was left. When the shape a fact travels in
  changes (a record gains a field, a value arrives whole where parts arrived before, a sample list
  stands where a directory stood), the functions that read it change their parameters to the new
  shape in the same change. A caller adapting a new-shape value to an old-shape parameter, by
  splitting a path into a directory and a stem, listing a directory to rebuild a sample list it
  already holds, or passing a config key where the record carries the value, keeps the old shape
  alive at that call and leaves the signature claiming it is the real one. Enumerate those
  callers; only a caller that genuinely holds the old shape keeps a parameter for it.
- A test that guards a fix is observed failing without the fix. Say when you have seen it fail and
  say when you have not. Every admits-valid-work test constructs its input through the platform's
  own producer.
- Before changing how a shared fact is spelled, produced or read, enumerate every producer of it
  and every reader of it, and state that list before the change rather than after. The rule that
  covers deleting something shared covers changing one: a convention changed at the site that
  named the defect and not at its siblings leaves the fact spelled two ways, and a producer and a
  reader spelling one key differently is a rail that answers no leak where there is one.
- A shared fact ships an agreement test: two routes' own records compared against each other,
  not each against a fixture. The two routes share one implementation of the fact (CLAUDE.md's
  "call one from the other"); the test guards that they still agree, and never licenses a second
  implementation to stay. Guard tests cannot stand in for it. A guard test proves the site it
  guards, so a suite of them stays green over exactly this defect, which is how it survives to be
  found by a reader instead of by the suite.
- Gates before reporting a change done: `ruff check packages tests tools`, `mypy`, and the change's
  own test files. Run the full suite on both storage backends when you touch the store seam or a
  reader's contract. Never report a gate before its slowest part finishes; green means no detected
  breakage, never correctness. A test touching the filesystem outside `tmp_path` is the first
  reread on a break that only shows in CI.
- A green suite is evidence only about the defects its fixtures can distinguish. Before trusting it
  as proof a change was safe, check that some fixture could have come out differently.
- The production mypy gate is the full one; only `tests` keeps grandfathered codes, in `mypy.ini`.
- Commits: one concern each, in dependency order, LF endings, messages stating the standing
  constraint the change installs, never a narrative of the session that made it.
- Match the weight of the process to the change. Reach for a worktree, a second reader or another
  model family when the change is genuinely hard to get right, not as a standing ritual, and say
  why you reached for it.
- A second model family's verdict is not advisory. When two families agree against your own
  position, conform to them or bring the split to the owner; the outvoted side never lands on the
  adjudicator's own authority.
- One fact has one spelling. Whenever a change touches a fact, ask what else states it: a second
  key, field, flag or parameter that must agree with the first is a derived spelling, whether or
  not this change introduced it, and it is deleted rather than kept, reconciled or asked which one
  governs. Delete it at the moment the surface changes, and conform every test config, sample
  project and knowledge sentence that carried it; those are sample data the work rewrites as it
  goes, never an interface whose current shape constrains the change. The commonest case is a
  record that can now carry a fact per item: the parameter that carried it once for the whole run
  goes, since its single value is usually the limit itself, and moving that value onto the new
  record rebuilds the limit one level down. A fact that cannot be re-derived and has nowhere yet
  to live is scheduled work, never a spelling left in place.

## Commands

```bash
conda activate tcip-agent          # Python 3.12; torch installs CUDA by default, runs without a GPU
pytest tests/ -n 4 --tb=short --timeout=300 -q
ruff check packages tests tools
mypy                               # roots from mypy.ini, run from the repo root
python tools/list_tools.py         # the MCP tool list (never hardcode counts in docs)
npm --prefix packages/tcip-web/frontend run build   # lint, typecheck and test take the same prefix
python -m tcip_web                 # backend plus built UI at http://127.0.0.1:8765
tcip export-store <root>           # a root's database-held records back out as files
tcip adopt-store <root>            # a root's loose record files into its database
```

Every process binds one storage backend at its entry point; an unset environment or
`TCIP_STORE_BACKEND=sqlite` binds the database (`<root>/.tcip/store.db`), `TCIP_STORE_BACKEND=file`
the loose-file layout, any other value refuses. `tests/test_store_contract.py` runs on both in one
run; the rest runs on whichever is bound, so run `pytest tests/` both ways when you touch the seam.
A root with loose records is refused by the database backend until `tcip adopt-store` conforms it.
The MCP server auto-launches from `.mcp.json`; a stale tool index means restart the client. Durable
state resolves via `$TCIP_STATE_ROOT`, pinned at startup by the web backend and every MCP server.

## Conventions

- Lazy-import torch and torchvision inside function bodies, so MCP startup stays fast.
- A one-off script the agent writes for one project lives with that project, in the project's own
  directory, never in this repository. A standing operator capability is a console command. Add an
  MCP tool only for an audit seam, long-running infrastructure, or domain knowledge the agent
  lacks.
- Crop traits are controlled vocabulary in `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`;
  verify there before asserting.
- Every piece of shipped prose (comments, docstrings, log and UI strings, test names, file names,
  scripts, README, skills, package `CLAUDE.md`s) is for whoever reads it next, never a changelog of
  the session that wrote it: no tracking labels, no inline decision dates, no bold or all-caps
  emphasis, no em dashes. If nothing survives once that framing is stripped, write nothing.
- `docs/` and `.claude/` are local, gitignored dev tooling, except the generated skills under
  `.claude/skills/`, which are tracked.
