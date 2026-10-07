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

## Phase 1a: TOI catalog and EDA

**Astrophysics**
- A TOI (TESS Object of Interest) is a flagged signal, not a confirmed planet.
- Dispositions: CP/KP (confirmed/known planet), PC/APC (still a candidate), FP/FA (false positive/alarm).
- Epochs are in BJD (Barycentric Julian Date), a time standard corrected for Earth's motion.

**ML concepts**
- Labels are vetting decisions, not ground truth, so the model imitates a vetting process.
- Selection bias: only ~35% of TOIs are resolved (CP/KP/FP/FA), and resolved ones skew toward easy cases. Metrics on this set will be optimistic for new signals.
- Single-feature ROC AUC (0.5 = no information) is a quick shortcut/leakage check. Here: period 0.64, depth 0.56, duration 0.53, tmag 0.49. Nothing is a giveaway; period is a mild one (false positives cluster below ~1 day).
- Group by star (TIC ID) when splitting: 233 stars have multiple TOIs.

**Engineering decisions**
- Fetch the catalog from the archive's TAP endpoint with pandas: scripted and reproducible, no extra dependency. Cached under `data/raw/`.
- Keep cleaning (`clean_toi_catalog`) a pure function, separate from the download, so it is testable without the network.
- PC/APC get a NaN label, not 0 or 1: they are unresolved, and forcing a label would inject noise. They stay in the table as an unlabeled set for error analysis later.

**Debugging lessons**
- The bare TAP URL returns `Input 'query' is blank`. Only the full URL from `build_query_url()` is a valid request.

**Open questions**
- Is the period skew real physics (eclipsing binaries and systematics at short periods) or a vetting artifact?
- How should single- versus multi-sector observations of a star be handled? (Phase 1b.)
