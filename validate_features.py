"""Reproduce the feature ablation with the official six monthly cutoffs.

Run from the starter repository root after copying this file and features.py:
    python validate_features.py
Uses data.py from the official starter and never tunes model parameters.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import data
import features


def original_features(homes, track):
    # Exact feature definitions from the supplied gradient_boosting_results.zip.
    parts = homes.house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = (low + high) / 2
    for dest, source in [("bedrooms", "bedrooms"), ("washrooms", "washrooms"),
                         ("parking", "parking_total"), ("latitude", "latitude"),
                         ("longitude", "longitude")]:
        X[dest] = homes[source]
    X["months"] = (homes.date_listed - pd.Timestamp("2021-01-01")).dt.days / 30.44
    for kind in features.PROPERTY_TYPES:
        X["is_" + kind] = (homes.house_type_name == kind).astype(int)
    if track == "list_aware":
        X["list_price"] = homes.list_price
    return X


def make_X(homes, track, variant):
    safe = data.hide_answers(homes, track)
    if variant == "original":
        return original_features(safe, track)
    X = (features.list_aware_features(safe) if track == "list_aware"
         else features.off_market_features(safe))
    if variant == "structured":
        X = X.drop(columns=[c for c in X if c.startswith("text_") or c == "remarks_missing"])
    return X


def score(rows):
    error = 100 * (rows.predicted_price - rows.sold_price) / rows.sold_price
    return {"n": int(len(rows)), "mdape": float(error.abs().median()),
            "mape": float(error.abs().mean()), "median_signed_error": float(error.median())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["original", "structured", "full"],
                        choices=["original", "structured", "full"])
    parser.add_argument("--out", default="feature_results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    listings = data.load("listings")
    months = pd.period_range("2025-12", "2026-05", freq="M")
    all_predictions = []
    # Static, label-free transformations may be cached; model fitting remains
    # strictly confined to the earlier sales in each fold.
    for variant in args.variants:
        for track in ["off_market", "list_aware"]:
            X = make_X(listings, track, variant)
            for month in months:
                start = time.monotonic()
                train = listings.date_sold < month.start_time
                valid = listings.date_sold.dt.to_period("M") == month
                with threadpool_limits(limits=4):
                    model = HistGradientBoostingRegressor(random_state=0)
                    model.fit(X.loc[train], np.log(listings.loc[train, "sold_price"]))
                    pred = np.round(np.exp(model.predict(X.loc[valid])))
                rows = listings.loc[valid, ["ml_num", "sold_price", "house_type_name", "community"]].copy()
                rows["month"] = str(month)
                rows["track"] = track
                rows["variant"] = variant
                rows["predicted_price"] = pred
                all_predictions.append(rows)
                rows.to_csv(out / f"predictions_{variant}_{track}_{month}.csv", index=False)
                print(f"{variant:10s} {track:10s} {month}: {score(rows)} [{time.monotonic()-start:.1f}s]", flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out / "all_predictions.csv", index=False)
    summary = []
    for (variant, track), rows in predictions.groupby(["variant", "track"]):
        summary.append({"variant": variant, "track": track, **score(rows)})
    pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)
    groups = ["month", "house_type_name", "community"]
    predictions["price_decile"] = predictions.groupby(["variant", "track"]).sold_price.transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1)
    for dimension in groups + ["price_decile"]:
        scores = []
        for keys, rows in predictions.groupby(["variant", "track", dimension], observed=True):
            scores.append(dict(zip(["variant", "track", dimension], keys), **score(rows)))
        pd.DataFrame(scores).to_csv(out / f"error_by_{dimension}.csv", index=False)
    (out / "experiment.json").write_text(json.dumps({
        "validation_months": [str(m) for m in months],
        "model": "HistGradientBoostingRegressor(random_state=0), all other parameters default",
        "round_predictions": True, "tuning_performed": False,
        "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                     "sklearn": __import__("sklearn").__version__},
        "feature_count": {track: make_X(listings.iloc[:2], track, "full").shape[1]
                          for track in ["off_market", "list_aware"]},
        "category_vocabulary_cutoff": "date_sold < 2025-12-01",
        "warning": "Ablation uses the same validation window as prior team tuning; not an untouched holdout."
    }, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
