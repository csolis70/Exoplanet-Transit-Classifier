# LEARNINGS

Study guide, updated after each phase.

## Phase 0: Repository setup

**Engineering decisions**
- `src/` layout: tests import the installed package, not stray local files.
- `pyproject.toml` declares dependencies and tooling config in one place.
- `data/` and `models/` are git-ignored: large, reproducible artifacts don't belong in Git.
- Virtual environment kept outside OneDrive to avoid syncing thousands of small files.

**Open questions**
- Why is a random split of light-curve segments a leakage risk? (Revisit in Phase 3.)
