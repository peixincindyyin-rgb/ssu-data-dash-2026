"""Numerical and temporal checks for the saved analysis; run after analyze_errors.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_errors import load_and_check, conformal_quantile, interval_backtest

root = Path(__file__).resolve().parent
df = load_and_check(root/"inputs/validation_predictions.csv", root/"inputs/validation_metadata.csv")
original = json.loads((root/"inputs/original_validation_report.json").read_text())
for track, g in df.groupby("track"):
    absolute = 100 * abs(g.predicted_price-g.sold_price) / g.sold_price
    assert round(float(absolute.mean()), 2) == original[track]["mape"]
    assert round(float(absolute.median()), 2) == original[track]["mdape"]
    assert len(g) == original[track]["n"]
    for column in ["house_type_name", "community", "municipality", "month", "sold_price_decile"]:
        table = pd.read_csv(root/f"tables/by_{column}.csv")
        assert table.loc[table.track == track, "n"].sum() == len(g)

intervals = pd.read_csv(root/"tables/interval_rows.csv")
eligible = intervals[intervals.eligible]
assert (eligible.lower <= eligible.upper).all()
assert (eligible.lower > 0).all()
assert (pd.to_datetime(eligible.max_calibration_sale_date) < pd.to_datetime(eligible.calibration_cutoff)).all()
for r in eligible.sample(min(60,len(eligible)), random_state=7).itertuples():
    pool = df[(df.track == r.track) & (df.date_sold < pd.Timestamp(r.calibration_cutoff))]
    scores = np.sort(np.abs(np.log(pool.sold_price/pool.predicted_price)))
    k = int(np.ceil((len(scores)+1)*.9))
    assert len(pool) == r.n_calibration
    assert np.isclose(scores[k-1], r.q_abs_log_error)
    assert np.isclose(r.predicted_price*np.exp(-scores[k-1]),r.lower)
    assert np.isclose(r.predicted_price*np.exp(scores[k-1]),r.upper)
    assert bool(r.covered) == (r.lower <= r.sold_price <= r.upper)

# Future labels must not affect earlier interval boundaries.
base = interval_backtest(df)
changed = df.copy()
changed.loc[changed.month == "2026-05", "sold_price"] *= 100
mutated = interval_backtest(changed)
assert np.allclose(base.lower,mutated.lower,equal_nan=True)
assert np.allclose(base.upper,mutated.upper,equal_nan=True)
assert conformal_quantile(np.arange(1,10),.1) == 9
assert np.isinf(conformal_quantile([1],.1))
assert (intervals[intervals.month == "2025-12"].n_calibration == 0).all()
result = {"passed": True, "checks": ["Original point metrics reproduced to two decimals",
          "Every subgroup table sums to its track total", "All interval dates strictly historical",
          "60 independently recomputed order statistics and interval bounds",
          "Changing May outcomes does not change any interval boundaries",
          "Small-calibration finite-sample boundary and December warmup"]}
(root/"tables/verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
