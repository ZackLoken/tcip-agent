## What this changes and why

## Checklist

- [ ] The change's own test files pass, each run to its end; CI runs the full suite
- [ ] The `ruff` lines of CLAUDE.md's Commands block pass
- [ ] `mypy` passes
- [ ] No tool count is written in a doc, comment or commit message (`python tools/list_tools.py`
      prints the registry)
- [ ] The frontend gate in CLAUDE.md's Commands block passes, if a frontend file changed
- [ ] A new or changed test constructs its input through the platform's own producer, not a
      hand-built fixture standing in for one
- [ ] Each guard's fail-before verdict is listed above
      (`python tools/prove_test_fails_before.py <testfile> -k <expr>`)
- [ ] If this touches a persisted format, a refusal, an operating-point stamp, or a delivery gate: a
      design issue was opened, discussed and read by another model family before this pull
      request, by two for a stored shape
- [ ] Commits are one concern each, in dependency order, with LF line endings, and each message
      states the standing constraint the change installs
- [ ] The line delta by area is stated above (`python tools/line_delta.py <repo> main..`), and the
      mechanism touched is left with less to maintain: none of the machinery kept beside its
      replacement that CONTRIBUTING.md names
- [ ] The code reviewer's and the test engineer's reads are recorded here with their verdicts
- [ ] No new or renamed crop trait vocabulary outside `crops.yml` and
      `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`
