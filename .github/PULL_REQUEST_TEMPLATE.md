## What this changes and why

## Checklist

- [ ] The change's own test files pass, each run to its end; CI runs the full suite
- [ ] The `ruff` line of CLAUDE.md's Commands block passes
- [ ] `mypy` passes
- [ ] `python tools/list_tools.py` was not hardcoded anywhere as a count in a doc, comment, or
      commit message
- [ ] The frontend gate in CLAUDE.md's Commands block passes, if a frontend file changed
- [ ] A new or changed test constructs its input through the platform's own producer, not a
      hand-built fixture standing in for one
- [ ] A test that guards a fix was shown failing without it
      (`python tools/prove_test_fails_before.py <testfile> -k <expr>`, verdict `GUARDS`)
- [ ] If this touches a persisted format, a refusal, an operating-point stamp, or a delivery gate: a
      design issue was opened and discussed before this pull request
- [ ] Commits are one concern each, in dependency order, with LF line endings, and each message
      states the standing constraint the change installs
- [ ] The line delta by area is stated above (`python tools/line_delta.py <repo> main..`), and the
      mechanism touched is left with less to maintain: no second implementation, no hand-written
      version of what a library provides, no vocabulary declared twice
- [ ] The code reviewer's and the test engineer's reads are recorded here with their verdicts
- [ ] No new or renamed crop trait vocabulary outside `crops.yml` and
      `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`
