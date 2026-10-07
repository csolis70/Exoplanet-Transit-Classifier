"""Download all SPOC sectors for a small, class-balanced pilot sample of labeled stars."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transit_classifier.catalog import clean_toi_catalog, fetch_toi_catalog
from transit_classifier.lightcurves import NoLightCurveError, download_star
from transit_classifier.sampling import select_pilot_sample

logger = logging.getLogger("download_pilot")

MANIFEST_PATH = Path("data/interim/pilot_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sample = select_pilot_sample(
        clean_toi_catalog(fetch_toi_catalog()), n_per_class=args.n_per_class, seed=args.seed
    )
    statuses, n_sectors = [], []
    for row in sample.itertuples():
        try:
            n = len(download_star(int(row.tic_id)))
            statuses.append("ok")
            n_sectors.append(n)
        except NoLightCurveError:
            logger.warning("TIC %d: no SPOC 2-minute data, skipping", row.tic_id)
            statuses.append("no_data")
            n_sectors.append(0)
        except Exception:
            logger.exception("TIC %d: download failed", row.tic_id)
            statuses.append("error")
            n_sectors.append(0)

    manifest = sample.assign(status=statuses, n_sectors=n_sectors)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    logger.info("Wrote %s\n%s", MANIFEST_PATH, manifest.groupby(["label", "status"]).size())


if __name__ == "__main__":
    main()
