# Feature engineering: measured results

Completed 2026-11-06. Evaluation: official walk-forward cutoffs, December 2025-May 2026, **10,069 homes per track**. All models use the same default HistGradientBoostingRegressor, random_state=0, and log sold-price target. No hyperparameter tuning was performed.

The full-feature experiment was independently rerun through the **unchanged official `python run.py validate`**. All **20,138 track/home predictions match exactly** between that command and the ablation runner. See `results/official_runner_equivalence.json`.

## Controlled feature comparison

| Features | Off-market MdAPE | Off-market MAPE | List-aware MdAPE | List-aware MAPE |
|---|---:|---:|---:|---:|
| Original supplied GB features | 14.87% | 19.23% | 4.61% | 5.82% |
| New structured features, without text | 10.34% | 13.26% | 4.42% | 5.55% |
| New structured + text features | 9.95% | 12.53% | 4.33% | 5.49% |

Off-market MdAPE fell by **4.92 percentage points (33.1% relative)**. List-aware MdAPE fell by **0.28 percentage points (6.2% relative)**. The rerun original-feature results match the uploaded GB report to two decimals.

The grouped structural/location changes account for most of the measured off-market gain. Adding text to that bundle reduces off-market MdAPE from 10.34% to 9.95% and list-aware MdAPE from 4.42% to 4.33%. This ablation does not isolate the individual effect of community, garage, seasonality or any single text term.

## Monthly full-feature results

| Month | Homes per track | Off-market MdAPE | List-aware MdAPE |
|---|---:|---:|---:|
| 2025-12 | 1,238 | 10.55% | 4.18% |
| 2026-01 | 1,058 | 9.90% | 4.33% |
| 2026-02 | 1,442 | 10.04% | 4.36% |
| 2026-03 | 1,816 | 9.56% | 4.29% |
| 2026-04 | 2,224 | 9.77% | 4.24% |
| 2026-05 | 2,291 | 10.01% | 4.52% |

The full feature set improves off-market MdAPE against the original in all six months. For list-aware, five months improve; January increases from 4.22% to 4.33%. Aggregate improvement is not uniform improvement.

## Findings for Member 4: error analysis

- Off-market remains weaker at price extremes. The cheapest decile has 13.00% MdAPE and +12.90% median signed error; the most expensive has 13.30% MdAPE and -8.11% median signed error. This is consistent with predictions pulling extreme homes toward more typical values; it is a pattern, not proof of a causal mechanism.
- Among large off-market groups, condo townhouses have 8.39% MdAPE (n=616); detached and semi-detached homes have 10.49% (n=3,123) and 10.54% (n=1,034). Co-op apartments have 12.20%, but only 48 examples, so avoid a strong ranking claim.
- List-aware condo apartments have 3.64% MdAPE versus 5.58% for detached homes. Asking-price information narrows the overall error, but uncertainty varies by home type.
- December off-market median signed error is +5.16%, compared with near-zero aggregate bias for some later months. List-aware predictions have an overall +1.14% median signed error. Bias correction is a possible future experiment, not applied in this handoff.
- Error breakdowns are provided by month, property type, community and sold-price decile. Community breakdowns include sample counts; avoid highlighting extreme errors from very small groups.
- Price-decile error analysis uses sold prices for diagnosis only, after prediction. Sold-price deciles are never inputs to the model.

## Handoff to Member 3

Replace only `features.py` in the team repository; preserve the tuned `models.py`. Keep pandas categorical columns and use the official scikit-learn 1.7.2. Use `categorical_features="from_dtype"` and `max_bins >= 145` if explicitly specifying these parameters. Then rerun the six-month validation before reporting the tuned/new-feature combination.

The bundled models.py is the supplied default model used for this controlled comparison. The eight-round tuned model/notebook was not supplied. Its reported 12.139% / 4.398% tuning-period results were not reproduced or combined with this feature set here.

## Validation notes

Results are from the December 2025–May 2026 walk-forward validation using the default Gradient Boosting model. The new features still need to be evaluated with Member 3’s tuned parameters.
