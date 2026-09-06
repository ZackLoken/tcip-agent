## What this changes and why

## Checklist

- [ ] `pytest tests/` passes (both storage backends, if this touches the storage seam or a
      change's own new/changed test files)
- [ ] `ruff check .` passes
- [ ] `mypy` passes
- [ ] `python tools/list_tools.py` was not hardcoded anywhere as a count in a doc, comment,
      or commit message
- [ ] `python tools/gate_baseline.py --out <dir>` passes, or CI ran the equivalent
- [ ] Frontend gate (`npm run format:check && npm run lint && npm run typecheck && npm test &&
      npm run build`) run if a frontend file changed
- [ ] A new or changed test constructs its input through the platform's own producer, not a
      hand-built fixture standing in for one
- [ ] A test that guards a fix was shown failing without it
      (`python tools/prove_test_fails_before.py <testfile> -k <expr>`, verdict `GUARDS`)
- [ ] If this touches a persisted format, a refusal, an operating-point stamp, or a delivery
      gate: a design issue was opened and discussed before this pull request
- [ ] If this touches a store's on-disk shape: `frozen-formats.json` was regenerated
      (`python tools/generate_frozen_manifest.py`) and `tests/test_frozen_manifest.py` passes
- [ ] Commits are one concern each, in dependency order, with LF line endings, and each message
      states the standing constraint the change installs
- [ ] No new or renamed crop trait vocabulary outside `crops.yml` and
      `packages/tcip-mcp/src/tcip_mcp/knowledge/crops/`
