"""Choose small, leakage-aware samples of the labeled TOI catalog."""

from __future__ import annotations

import pandas as pd


def select_pilot_sample(catalog: pd.DataFrame, n_per_class: int = 25, seed: int = 0) -> pd.DataFrame:
    """Return ``n_per_class`` labeled TOIs per class, at most one TOI per star.

    Stars whose labeled TOIs disagree (e.g. one CP and one FP) are excluded, and each star appears
    once, so the pilot has no star-level duplicates. Sampling is seeded for reproducibility.
    """
    labeled = catalog[catalog["label"].notna()]
    per_star = labeled.groupby("tic_id")["label"].nunique()
    consistent = labeled[labeled["tic_id"].isin(per_star[per_star == 1].index)]
    one_per_star = consistent.sort_values("toi").drop_duplicates("tic_id")

    parts = []
    for label, group in one_per_star.groupby("label"):
        if len(group) < n_per_class:
            raise ValueError(f"Only {len(group)} stars available for label {label}")
        parts.append(group.sample(n=n_per_class, random_state=seed))
    return pd.concat(parts).sort_values("toi").reset_index(drop=True)
