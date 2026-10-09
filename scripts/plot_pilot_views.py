"""Fold the downloaded pilot stars at their catalog ephemerides and plot global/local views."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from transit_classifier.lightcurves import download_star  # noqa: E402
from transit_classifier.preprocess import BTJD_OFFSET, make_views, stitch  # noqa: E402

logger = logging.getLogger("plot_pilot_views")

MANIFEST_PATH = Path("data/interim/pilot_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("docs/pilot_views.png"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["status"] == "ok"]
    rows = pd.concat(
        [manifest[manifest["label"] == lab].head(args.n_per_class) for lab in (1.0, 0.0)]
    )

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 2.2 * len(rows)), squeeze=False)
    for ax_row, row in zip(axes, rows.itertuples()):
        sectors = [(lc.time.value, lc.flux.value) for lc in download_star(int(row.tic_id))]
        time, flux = stitch(sectors)
        try:
            g, loc = make_views(
                time,
                flux,
                period=row.period_days,
                epoch=row.epoch_bjd - BTJD_OFFSET,
                duration=row.duration_hours / 24,
            )
        except ValueError:
            logger.exception("TIC %d: could not build views", row.tic_id)
            continue
        tag = "candidate-positive (CP/KP)" if row.label == 1 else "negative (FP/FA)"
        ax_row[0].plot(np.linspace(-0.5, 0.5, len(g)), g * 1e6, ".-", ms=3, lw=0.5)
        ax_row[0].set_ylabel(f"TOI {row.toi}\n{row.disposition}, P={row.period_days:.2f} d\nppm")
        ax_row[1].plot(np.linspace(-2, 2, len(loc)), loc * 1e6, ".-", ms=3, lw=0.5)
        ax_row[1].set_title(f"{tag}, depth(cat)={row.depth_ppm:.0f} ppm", fontsize=8)
    axes[0, 0].set_title("global view (phase / period)", fontsize=8)
    axes[-1, 0].set_xlabel("phase")
    axes[-1, 1].set_xlabel("time from mid-transit (transit durations)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    logger.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
