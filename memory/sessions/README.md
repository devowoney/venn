# memory/sessions/ — one file per working session

Convention (user request, 2026-08-28): each working session gets its own dated record here, so
the trail survives a shutdown and a new session can resume without re-deriving anything.

- **File name:** `YYYY-MM-DD_short-topic.md` (start date of the session).
- **Contents:** what was asked, what was built, what was measured, what broke, decisions/findings
  produced (cross-referenced to `decisions.md` D-xxx and `findings.md` F-x), files touched,
  commands to reproduce, and the open thread to pick up next.
- **Relationship to the other memory files:** `progress.md` stays the short chronological log,
  `decisions.md` holds the D-xxx rationale, `findings.md` holds the F-x evidence. These session
  files are the long-form narrative that ties them together — including the dead ends, which the
  other three deliberately do not record.
- Raw Claude Code transcripts (JSONL) live outside the repo in
  `~/.claude/projects/-home-sysadmin-jlee-venn/<session-uuid>.jsonl`; each record below names its
  own transcript file.

Note: `memory/` was originally git-ignored (user's `.gitignore`, "AI artifacts"), which is why a tracked
`architecture/` copy once existed. As of 2026-09-28 `memory/` is the tracked home of the project record and
`architecture/` is retired (merged into `memory/`; SOPs in `memory/sop/`).

| session | topic |
|---|---|
| [2026-08-26](2026-08-26_v0-objective-and-generator-rev2.md) | v0 encoder objective made to work (D-016…D-020) + generator rev2 (D-021) |
| [2026-08-29](2026-08-29_latent-predictor-module2.md) | module 2 latent predictor (D-022, D-023, F-9…F-12 module-2 track, F-18) |
| [2026-08-31](2026-08-31_stationary-observer-and-timescale-ladder.md) | stationary observer, timescale ladder (D-024…D-026) |
| [2026-09-01](2026-09-01_stationary-observer-merge-handoff.md) | stationary-observer merge handoff (was `architecture/README.md`) |
| [2026-09-28](2026-09-28_predictor-memory-sight-theory.md) | module 2: trajectory as memory, "sight", W8 → one model (was `03e`) |
