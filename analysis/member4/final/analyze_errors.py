"""Reproduce error diagnostics and strictly historical interval backtests.

Usage from this folder:
  python analyze_errors.py
  python analyze_errors.py --predictions path/to/validation_predictions.csv \
      --metadata path/to/listings.parquet --out new_model_analysis

Metadata is used only for diagnostics and availability dates, never to train a model.
This script evaluates supplied predictions; it cannot certify how they were trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = ["off_market", "list_aware"]
COLORS = {"off_market": "#28649A", "list_aware": "#C26435"}
LABELS = {"off_market": "Off-market", "list_aware": "List-aware"}


def metrics(g):
    e = g["signed_error_pct"].to_numpy()
    a = np.abs(e)
    return {"n": len(g), "mdape_pct": np.median(a), "mape_pct": np.mean(a),
            "median_signed_pct": np.median(e), "p90_ape_pct": np.quantile(a, .9),
            "mae_dollars": np.mean(np.abs(g.predicted_price - g.sold_price)),
            "overprediction_pct": 100 * np.mean(e > 0),
            "within_10pct": 100 * np.mean(a <= 10),
            "within_20pct": 100 * np.mean(a <= 20),
            "above_50pct": 100 * np.mean(a > 50)}


def grouped(df, column):
    rows = []
    for (track, group), g in df.groupby(["track", column], observed=True, dropna=False):
        rows.append({"track": track, column: group, **metrics(g),
                     "small_group_lt50": len(g) < 50})
    return pd.DataFrame(rows)


def bootstrap_metrics(g, repeats=1000, seed=42):
    """Two descriptive resampling sensitivities; only six month clusters exist."""
    rng = np.random.default_rng(seed)
    a = g.ape_pct.to_numpy()
    blocks = [x.ape_pct.to_numpy() for _, x in g.groupby("month")]
    result = {}
    for method in ["iid_homes", "month_clusters"]:
        samples = []
        for _ in range(repeats):
            if method == "iid_homes":
                x = rng.choice(a, len(a), replace=True)
            else:
                ix = rng.integers(0, len(blocks), size=len(blocks))
                x = np.concatenate([blocks[i] for i in ix])
            samples.append([np.median(x), np.mean(x)])
        low, high = np.quantile(samples, [.025, .975], axis=0)
        result[method] = {"mdape_95pct": [float(low[0]), float(high[0])],
                          "mape_95pct": [float(low[1]), float(high[1])]}
    return result


def conformal_quantile(scores, alpha):
    """Finite-sample order statistic, without a time-series coverage guarantee."""
    scores = np.asarray(scores, dtype=float)
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if len(scores) == 0 or k > len(scores):
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def interval_backtest(df, alpha=.10, min_calibration=100):
    """Use residuals from prior folds AND sales strictly before listing date.

    cutoff = min(evaluation month start, date_listed).
    Include a calibration residual only if its sale date is strictly before cutoff.
    No same-month outcome is used, even if it sold earlier than the target home.
    Each track has its own pool; the point predictions are unchanged.
    """
    result = []
    for track, data in df.groupby("track"):
        pool = data.sort_values("date_sold").copy()
        available = pool.date_sold.to_numpy(dtype="datetime64[ns]")
        scores = np.abs(np.log(pool.sold_price / pool.predicted_price)).to_numpy()
        cache = {}
        for row in data.itertuples():
            cutoff = min(pd.Timestamp(row.month + "-01"), row.date_listed)
            n = int(np.searchsorted(available, np.datetime64(cutoff), side="left"))
            eligible = n >= min_calibration
            if eligible:
                if n not in cache:
                    cache[n] = conformal_quantile(scores[:n], alpha)
                q = cache[n]
                lower, upper = row.predicted_price * np.exp(-q), row.predicted_price * np.exp(q)
                covered = lower <= row.sold_price <= upper
                max_sale = pool.date_sold.iloc[n-1]
                if not max_sale < cutoff:
                    raise AssertionError("Calibration availability check failed")
            else:
                q = lower = upper = np.nan
                covered, max_sale = np.nan, pd.NaT
            result.append({"ml_num": row.ml_num, "track": track, "month": row.month,
                           "house_type_name": row.house_type_name, "community": row.community,
                           "date_listed": row.date_listed, "calibration_cutoff": cutoff,
                           "max_calibration_sale_date": max_sale, "n_calibration": n,
                           "eligible": eligible, "q_abs_log_error": q,
                           "predicted_price": row.predicted_price, "sold_price": row.sold_price,
                           "lower": lower, "upper": upper, "covered": covered,
                           "width_pct_of_prediction": 100*(upper-lower)/row.predicted_price})
    return pd.DataFrame(result)


def interval_summary(intervals, group=None):
    keys = ["track"] + ([group] if group else [])
    rows = []
    for key, all_g in intervals.groupby(keys, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        g = all_g[all_g.eligible]
        rows.append({**dict(zip(keys, key)), "n_total": len(all_g), "n_evaluated": len(g),
                     "n_excluded": len(all_g)-len(g),
                     "coverage_pct": 100 * g.covered.astype(float).mean(),
                     "median_width_pct_of_prediction": g.width_pct_of_prediction.median(),
                     "median_width_dollars": (g.upper-g.lower).median(),
                     "miss_below_pct": 100*(g.sold_price < g.lower).mean() if len(g) else np.nan,
                     "miss_above_pct": 100*(g.sold_price > g.upper).mean() if len(g) else np.nan})
    return pd.DataFrame(rows)


def load_and_check(predictions, metadata):
    df = pd.read_csv(predictions, dtype={"ml_num": str, "month": str})
    req = {"ml_num", "track", "month", "house_type_name", "sold_price", "predicted_price"}
    if not req <= set(df):
        raise ValueError(f"Missing prediction columns: {sorted(req-set(df))}")
    if df[list(req)].isna().any().any() or df.duplicated(["track", "ml_num"]).any():
        raise ValueError("Null required values or duplicated (track, ml_num)")
    if set(df.track) != set(TRACKS):
        raise ValueError("Expected both off_market and list_aware tracks")
    for c in ["sold_price", "predicted_price"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not (np.isfinite(df[c]).all() and (df[c] > 0).all()):
            raise ValueError(f"{c} must be finite and positive")
    md = pd.read_parquet(metadata) if metadata.suffix == ".parquet" else pd.read_csv(metadata)
    wanted = ["ml_num", "date_listed", "date_sold", "community", "municipality",
              "sold_price", "house_type_name"]
    optional = [c for c in ["house_area", "bedrooms", "washrooms", "list_price", "lot_front", "lot_depth"] if c in md]
    md = md[wanted + optional].rename(columns={"sold_price": "source_sold_price",
                                               "house_type_name": "source_house_type"})
    df = df.merge(md, on="ml_num", how="left", validate="many_to_one", indicator=True)
    if not (df._merge == "both").all():
        raise ValueError("Some prediction IDs are missing from metadata")
    if not np.allclose(df.sold_price, df.source_sold_price, rtol=0, atol=.01):
        raise ValueError("Sold prices differ from official metadata")
    if not (df.house_type_name == df.source_house_type).all():
        raise ValueError("Property types differ from official metadata")
    for c in ["date_listed", "date_sold"]:
        df[c] = pd.to_datetime(df[c], errors="raise")
    if df[["date_listed", "date_sold"]].isna().any().any():
        raise ValueError("Missing dates prevent historical calibration")
    if not (df.date_sold.dt.strftime("%Y-%m") == df.month).all():
        raise ValueError("Validation month differs from sale month")
    if (df.date_listed > df.date_sold).any():
        raise ValueError("Listing date after sale date")
    df = df.drop(columns=["_merge", "source_sold_price", "source_house_type"])
    df["signed_error_pct"] = 100 * (df.predicted_price-df.sold_price)/df.sold_price
    df["ape_pct"] = df.signed_error_pct.abs()
    for c, name in [("sold_price", "sold_price_decile"), ("predicted_price", "predicted_price_decile")]:
        df[name] = df.groupby("track")[c].transform(
            lambda s: pd.qcut(s, 10, labels=False, duplicates="drop") + 1)
    return df


def make_figures(df, intervals, out, alpha=.10):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.dpi": 180})
    def finish(fig, name):
        fig.savefig(out / "figures" / (name + ".png"), bbox_inches="tight")
        fig.savefig(out / "figures" / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    types = df.house_type_name.value_counts().index.tolist()
    y = np.arange(len(types))
    for i, track in enumerate(TRACKS):
        g = df[df.track == track]
        typ = grouped(g, "house_type_name").set_index("house_type_name").reindex(types)
        axes[0, 0].barh(y+(i-.5)*.34, typ.mdape_pct, height=.32, color=COLORS[track], label=LABELS[track])
        dec = grouped(g, "sold_price_decile")
        axes[0, 1].plot(dec.sold_price_decile, dec.median_signed_pct, "o-", color=COLORS[track], label=LABELS[track])
        mon = grouped(g, "month")
        axes[1, 0].plot(mon.month.str[2:], mon.mdape_pct, "o-", color=COLORS[track], label=LABELS[track])
        a = np.sort(g.ape_pct.to_numpy())
        axes[1, 1].plot(a, 100*np.arange(1,len(a)+1)/len(a), color=COLORS[track], label=LABELS[track])
    counts = df[df.track == "off_market"].house_type_name.value_counts()
    axes[0, 0].set_yticks(y, [f"{t} (n={counts[t]:,})" for t in types])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(title="Error varies by property type", xlabel="MdAPE (%)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].axhline(0, color="#777777", lw=1)
    axes[0, 1].set(title="Signed error across sale-price deciles", xlabel="Actual sale-price decile (retrospective)", ylabel="Median signed error (%)", xticks=range(1,11))
    axes[1, 0].set(title="Monthly validation error", ylabel="MdAPE (%)", xlabel="Sale month")
    axes[1, 1].axhline(90, color="#777777", ls="--", lw=1)
    axes[1, 1].set(title="Share of homes within an error threshold", xlabel="Absolute percentage error threshold (%)", ylabel="Cumulative share (%)", xlim=(0,100), ylim=(0,102))
    for ax in axes.flat:
        ax.grid(axis="x" if ax == axes[0,0] else "y", alpha=.15)
    fig.suptitle("Final tuned model error analysis | Dec 2025 - May 2026", fontsize=16, fontweight="bold")
    finish(fig, "01_error_overview")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, track in zip(axes, TRACKS):
        s = grouped(df[df.track == track], "community").query("n >= 50").nlargest(8,"mdape_pct").sort_values("mdape_pct")
        ax.barh(s.community, s.mdape_pct, color=COLORS[track], height=.65)
        for k, row in enumerate(s.itertuples()):
            ax.text(row.mdape_pct+.3, k, f"{row.mdape_pct:.1f}% | n={row.n}", va="center", fontsize=9)
        ax.set_xlim(0, s.mdape_pct.max()*1.42)
        ax.set(title=LABELS[track], xlabel="MdAPE (%)")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Communities with the largest observed errors (n >= 50)\nExploratory ranking; property mix differs; synthetic names", fontsize=14, fontweight="bold")
    finish(fig, "02_community_errors")

    nominal = 100 * (1-alpha)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for track in TRACKS:
        ints = intervals[intervals.track == track]
        mon = interval_summary(ints, "month").query("n_evaluated > 0")
        axes[0].plot(mon.month.str[2:], mon.coverage_pct, "o-", color=COLORS[track], label=LABELS[track])
        axes[1].plot(mon.month.str[2:], mon.median_width_pct_of_prediction, "o-", color=COLORS[track])
        typ = interval_summary(ints,"house_type_name").set_index("house_type_name").reindex(types)
        axes[2].plot(typ.coverage_pct, y, "o", color=COLORS[track], label=LABELS[track])
    axes[0].axhline(nominal, color="#777777", ls="--", label=f"Nominal {nominal:g}%")
    axes[0].set(title="Observed coverage", ylabel="Coverage (%)", ylim=(75,100), xlabel="Evaluation sale month")
    axes[0].legend(fontsize=9,frameon=False,loc="lower left")
    axes[1].set(title="Interval width", ylabel="Median width / prediction (%)", xlabel="Evaluation sale month", ylim=(0,None))
    axes[2].axvline(nominal,color="#777777",ls="--")
    axes[2].set_yticks(y,types)
    axes[2].set(title="Coverage by property type", xlabel="Coverage (%)", xlim=(60,102))
    axes[2].invert_yaxis()
    for ax in axes: ax.grid(alpha=.15)
    fig.suptitle(f"Historical {nominal:g}% interval backtest | earlier outcomes only\nCalibration sales precede both the listing date and evaluation month", fontsize=14, fontweight="bold")
    finish(fig,"03_prediction_intervals")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=root/"inputs/validation_predictions.csv")
    parser.add_argument("--metadata", type=Path, default=root/"inputs/validation_metadata.csv")
    parser.add_argument("--out", type=Path, default=root)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--min-calibration", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if not 0 < args.alpha < 1 or args.min_calibration < 1 or args.bootstrap < 100:
        parser.error("alpha must be in (0,1), min-calibration >= 1, bootstrap >= 100")
    args.out = args.out.resolve()
    for folder in ["tables", "figures"]: (args.out/folder).mkdir(parents=True, exist_ok=True)
    df = load_and_check(args.predictions, args.metadata)
    summary = {t: {**metrics(df[df.track == t]),
                   "bootstrap": bootstrap_metrics(df[df.track == t], args.bootstrap)} for t in TRACKS}
    (args.out/"tables/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{ "track": t, **metrics(df[df.track == t])} for t in TRACKS]).to_csv(args.out/"tables/overall_metrics.csv",index=False)
    for column in ["house_type_name", "month", "sold_price_decile", "predicted_price_decile", "community", "municipality"]:
        grouped(df,column).to_csv(args.out/f"tables/by_{column}.csv",index=False)
    df.to_csv(args.out/"tables/row_errors.csv",index=False)
    df.sort_values("ape_pct",ascending=False).groupby("track",sort=False).head(20).to_csv(args.out/"tables/largest_40_errors.csv",index=False)
    intervals = interval_backtest(df, args.alpha, args.min_calibration)
    intervals.to_csv(args.out/"tables/interval_rows.csv",index=False)
    for group in [None,"month","house_type_name","community"]:
        interval_summary(intervals,group).to_csv(args.out/f"tables/interval_{group or 'overall'}.csv",index=False)
    audit = {"prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
             "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
             "rows": len(df), "unique_homes": df.ml_num.nunique(), "months": sorted(df.month.unique().tolist()),
             "positive_finite_prices": True, "unique_track_id": True, "metadata_match": True,
             "dates_match": True, "alpha": args.alpha, "min_calibration": args.min_calibration,
             "bootstrap_repeats": args.bootstrap, "bootstrap_seed": 42,
             "interval_rule": "calibration sale < min(target listing date, target sale-month start)",
             "calibration_strictly_historical": True,
             "model_training_code_audited": False,
             "note": "Training provenance cannot be established from a predictions CSV. No hidden-test data used."}
    (args.out/"tables/audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    make_figures(df,intervals,args.out,args.alpha)
    print(pd.read_csv(args.out/"tables/overall_metrics.csv").round(3).to_string(index=False))
    print(interval_summary(intervals).round(3).to_string(index=False))
    print(f"Wrote analysis to {args.out}")


if __name__ == "__main__":
    main()
