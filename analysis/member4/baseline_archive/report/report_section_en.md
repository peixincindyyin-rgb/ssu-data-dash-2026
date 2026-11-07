# Error analysis and uncertainty — baseline draft

**Scope:** This text describes the supplied baseline. Replace the results after evaluating the team's final model.

Across 10,069 walk-forward validation homes per track (December 2025–May 2026), off-market MdAPE/MAPE were 20.84%/27.43%, versus 7.26%/8.09% for list-aware. Median signed errors were +8.07% and +5.68%, indicating overprediction. Off-market errors were larger for detached homes (MdAPE 27.99%; n=3,123) than condo townhouses (16.39%; n=616). The cheapest and most expensive sale-price deciles had median signed errors of +31.44% and −36.40%. These outcome-defined groups are retrospective diagnostics, not deployable features or causal evidence.

An extreme off-market case sold for $844,000 but was predicted at $4,317,900. Reconstructing its ten comparable sales reproduced the prediction: their median area was 1,883.5 sq ft, compared with the target's 1,166 sq ft, and only three shared its community. This motivates testing size, community and recency adjustments; it does not establish their individual effects.

We also backtested nominal 90% intervals using historical absolute log residuals, separately by track. Calibration sales had to precede both the target's listing date and evaluation month; at least 100 residuals were required. On 8,045 eligible homes per track, coverage was 90.11% off-market and 89.36% list-aware. Median interval widths were 114.23% and 30.04% of the point prediction. Off-market detached coverage was only 80.28%, exposing uneven uncertainty despite near-target aggregate coverage. The remaining 2,024 homes per track lacked sufficient calibration history. Time drift and changing fitted models prevent us from claiming a strict conformal coverage guarantee. Future work should evaluate property-aware calibration on later data.

Suggested figure: selected panels from `01_error_overview.png` and `03_prediction_intervals.png`. All numerical claims above come from the supplied validation predictions and their official metadata join; no hidden-test outcomes were used.
