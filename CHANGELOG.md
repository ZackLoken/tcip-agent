# Changelog

Git history is the authority for what changed and when; this file does not restate it. It
records adopter-visible changes, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
form. No release has been tagged yet (see VERSIONING.md), so everything below is unreleased.

## Unreleased

### Added

- CONTRIBUTING.md: setup pointer, the gate a change must pass, and the rules a contributor
  meets (producer-built test inputs, fail-before proof for a guarding test, the frozen-format
  commitment, design review before code for a persisted-field or refusal change).
- CODE_OF_CONDUCT.md: the Contributor Covenant, version 2.1.
- A "Reporting a vulnerability" section in SECURITY.md: GitHub private vulnerability reporting
  as the primary channel, with an issue-titled-"security" fallback, and a stated acknowledgement
  window.
- Issue templates (`.github/ISSUE_TEMPLATE/`): a bug report, a measurement report for a
  delivered phenotype that is wrong or overclaims (asking for the delivered CSV's own
  provenance-tail columns), and a design proposal for a change touching a persisted format, a
  refusal, or a delivery gate.
- A pull request template (`.github/PULL_REQUEST_TEMPLATE.md`) checklist mirroring
  CONTRIBUTING.md's gates.
- VERSIONING.md and STABILITY.md: what the four packages' shared version number means, and what
  an adopter can rely on today (the frozen persisted formats, the MCP tool surface, the two
  storage backends, `tcip-annotation`'s public API, the general-capability scripts) versus what
  is explicitly not stable yet.
- A "From images to a first delivered number" walkthrough in README.md, naming the real tools
  and GUI tabs a first end-to-end run touches, and a "Developer tooling" note describing the
  maintainer's optional semantic-search server as machine-local configuration outside the
  tracked `.mcp.json`.
- `tests/test_scripts_readme_index.py`: every tracked file under `scripts/` is named in
  `scripts/README.md` as a backticked filename, so the index cannot silently drift behind the
  directory.
- A coverage test in `tests/test_resolution.py` asserting every constant in
  `pipelines/resolution.py`'s `VALIDATED_SHIPPABLE` appears in some `_DIMENSION_REFERENCES`
  tuple, the invariant the delivery gate's dimension table depends on.
- The `domain_knowledge` MCP tool, composing its client-visible description from the knowledge
  corpus at import time, and generated skill files under `.claude/skills/<name>/SKILL.md` and
  `.agents/skills/<name>/SKILL.md` plus a generated block in `AGENTS.md`, all rendered from that
  same corpus by `scripts/generate_harness_discovery.py`, so Claude Code, Codex, Antigravity and
  any other MCP client read the same domain knowledge through the surface each can reach.
- README's roadmap now names plant-tag identity (a QR or barcode physically tied to the plant)
  as future work for capture with no georeferencing, distinct from the geolocated-capture and
  orthomosaic paths already built.

### Changed

- `.mcp.json` now declares only the platform's own `tcip` MCP server; the maintainer's optional
  semantic-search server is configured per machine outside this tracked file.
- `scripts/README.md` now indexes every tracked file under `scripts/`.

### Removed

- `archive_project`, `import_project`, `scan_dataset`, `inspect_compute_resources`, and
  `render_failure_cases` are no longer registered MCP tools; each is now reached through its own
  `scripts/<name>.py` entry point instead.
