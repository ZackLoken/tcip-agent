## What this changes and why

## Checklist

- [ ] `pytest tests/` passes (or the change's own new/changed test files)
- [ ] `ruff check .` passes
- [ ] `mypy` passes
- [ ] `python tools/list_tools.py` was not hardcoded anywhere as a count in a doc, comment, or
      commit message
- [ ] `python tools/gate_baseline.py --out <dir>` passes, or CI ran the equivalent
- [ ] Frontend gate run if a frontend file changed: `npm run format:check`, `npm run lint`,
      `npm run typecheck`, `npm test` and `npm run build`
- [ ] A new or changed test constructs its input through the platform's own producer, not a
      hand-built fixture standing in for one
- [ ] A test that guards a fix was shown failing without it
      (`python tools/prove_test_fails_before.py <testfile> -k <expr>`, verdict `GUARDS`)
- [ ] If this touches a persisted format, a refusal, an operating-point stamp, or a delivery gate: a
      design issue was opened and discussed before this pull request
- [ ] Commits are one concern each, in dependency order, with LF line endings, and each message
      states the standing constraint the change installs
- [ ] No new or renamed crop trait vocabulary outside `crops.yml` and
      `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`
