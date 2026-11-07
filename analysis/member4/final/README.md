# Final tuned model: Member 4 error analysis

This replaces the earlier baseline-only analysis. It uses the team repository's final `validation_predictions.csv` at commit `f3c5aee2630d2c36d63c2c1f5748c572739d63f9`, plus official training-file metadata for the same validation homes. The repository's `models.py` has Member 3's Round 8 settings and `features.py` has Member 2's updated features. No model was retrained for this analysis.

**For slides and voice-over:** start with [`report/VIDEO_FINDINGS.md`](report/VIDEO_FINDINGS.md).

| Track | Homes | MdAPE | MAPE | 90th-percentile APE | Median signed error |
|---|---:|---:|---:|---:|---:|
| Off-market | 10,069 | 7.79% | 10.35% | 21.60% | +0.48% |
| List-aware | 10,069 | 4.23% | 5.36% | 11.39% | +0.72% |

The tracks use the same validation homes but answer different pricing questions. Off-market has no list price. Point metrics include all validation homes; the exploratory price-interval backtest includes 8,045 eligible homes per track. The other 2,024 lack adequate earlier validation residuals, including all of December. Interval coverage is 89.77% off-market and 88.55% list-aware, with substantially lower detached-home coverage (84.54% and 80.58%).

`tables/` has row-level errors, grouped metrics by month, property type, community, municipality, actual sale-price decile and predicted-price decile, plus historical interval results. The `small_group_lt50` column identifies thin groups. Actual sale-price deciles are post-sale diagnostics; they are not inputs to the model. Community rankings are exploratory and unadjusted for property mix.

## Reproduce

With Python and `numpy`, `pandas`, `matplotlib` installed, run:

```bash
python analyze_errors.py
```

The included `inputs/validation_metadata.csv` supplies only the validation homes' metadata and outcome date for the join and chronological interval calculation. `analyze_errors.py` never fits a model or alters point predictions. It requires positive finite prices, unique track/ID pairs, a match with official sold prices and property types, and sale dates that agree with validation months. The audit file records input hashes and the temporal calibration rule. Run the same analysis on later model predictions with `--predictions`, `--metadata`, and `--out`.

## Verification

- Recomputed MdAPE/MAPE agree with the final team's reported numbers to two decimals.
- Each track contains 10,069 rows covering December 2025-May 2026; metadata joins and dates passed the script's checks.
- The team's submission files have 3,974 off-market and 4,064 list-aware rows. The official `check_submission.py` returned `OK` for both against the official test input IDs.
- Interval calibration uses only validation homes whose sale date is strictly before both the target's listing date and its evaluation month. Point forecasts were not recalibrated. These intervals are an exploratory analysis, not part of either submission CSV.
- The same validation period was used for feature development and tuning, so it is a development estimate rather than an untouched final test. The hidden-test prices were not accessed.

Sources: [team repository](https://github.com/peixincindyyin-rgb/ssu-data-dash-2026), [official synthetic dataset](https://huggingface.co/datasets/Baliyang/SSU_data_dash), user-provided baseline archive, and the competition case handout. The interval construction borrows a quantile idea from [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511); time drift means its usual exchangeability guarantee is not claimed here.
