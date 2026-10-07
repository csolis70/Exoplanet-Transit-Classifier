# Exoplanet Transit Classifier

A machine learning project that analyzes stellar light curves from NASA's
[TESS](https://tess.mit.edu/) mission and predicts whether a light curve contains a
**likely exoplanet transit candidate**.

> **Terminology note:** this model flags *transit candidates*. A candidate signal can also be
> caused by eclipsing binaries, instrumental systematics, or stellar variability. The model does
> not confirm that a planet exists.

## Background

### The physics

When a planet passes in front of its star, the star's brightness dips slightly, usually by about
0.1% to 1%. TESS records a star's brightness every few minutes for weeks. That record is a *light
curve*. A planet shows up as a small dip that repeats at regular intervals.

### The problem

Thousands of stars produce dips, and most are not planets. Eclipsing binary stars, starspots, and
instrument glitches all make dips too. Astronomers have to sort real candidates from impostors,
and doing that by hand does not scale.

### The ML task

Given a light curve, predict whether the dip is a planet candidate or an impostor. This is binary
classification. Training labels come from the NASA TOI catalog, where astronomers have already
judged many signals (confirmed planets are positive, false positives are negative).

### Example: a planet vs. a false positive

![Raw and phase-folded light curves for a known planet (TOI 192.01) and a false positive
(TOI 1023.01)](docs/light_curve_comparison.png)

Top row: TOI 192.01, a known planet. The raw light curve (left) shows repeating dips about 1.2%
deep. Folding at the known period (right) stacks them into a clean, sharp, flat-bottomed dip.
Bottom row: TOI 1023.01, a false positive. Its folded dip is shallower, wider, and rounded.

Why this matters for the ML: the model's job is to learn the difference between these two
right-hand panels. That is why we fold and bin the data. Every example then becomes a short,
fixed-length shape the model can compare.

## Status

Early development. Done so far: repository scaffolding, TOI catalog download/cleaning, and a
catalog EDA notebook. Next: downloading light curves.

## Data source

Labels and signal parameters come from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) TESS Project Candidates (TOI)
table, fetched through its TAP service (`src/transit_classifier/catalog.py`). The catalog changes
over time, so results are tied to a snapshot: the current local snapshot was downloaded on
**2026-10-07** (8,148 TOIs, 2,801 with a CP/KP/FP/FA label).

## Planned pipeline

TESS data -> acquisition -> preprocessing -> feature extraction -> model -> inference service
-> REST API -> web interface

## Roadmap

- [ ] Data acquisition (TESS light curves + TOI catalog labels)
- [ ] Preprocessing (quality masks, detrending, phase folding)
- [ ] EDA and leakage-safe train/validation/test split (grouped by star)
- [ ] Baseline and feature-based models
- [ ] Error analysis; optional 1D CNN
- [ ] Inference pipeline and model persistence
- [ ] FastAPI service + Docker
- [ ] CI/CD and cloud deployment

## Setup

```bash
python -m venv <path-to-venv>
<path-to-venv>/Scripts/activate      # Windows
pip install -e ".[dev]"
pytest
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/transit_classifier/` | Importable package code |
| `tests/` | Automated tests |
| `notebooks/` | Exploration only; real logic lives in `src/` |
| `configs/` | Configuration files |
| `data/`, `models/` | Local data and trained artifacts (git-ignored) |
| `docs/` | Documentation |
| `LEARNINGS.md` | Study guide maintained as the project grows |
