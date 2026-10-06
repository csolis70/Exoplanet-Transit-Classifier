# Exoplanet Transit Classifier

A machine learning project that analyzes stellar light curves from NASA's
[TESS](https://tess.mit.edu/) mission and predicts whether a light curve contains a
**likely exoplanet transit candidate**.

> **Terminology note:** this model flags *transit candidates*. A candidate signal can also be
> caused by eclipsing binaries, instrumental systematics, or stellar variability. The model does
> not confirm that a planet exists.

## Status

Early development. Currently: repository scaffolding.

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
