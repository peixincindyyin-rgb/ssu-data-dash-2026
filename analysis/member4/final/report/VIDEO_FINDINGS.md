# Member 4 video findings - final tuned model

Source: `validation_predictions.csv` at team repository commit `f3c5aee2630d2c36d63c2c1f5748c572739d63f9`. All numbers below were recomputed from its 20,138 official walk-forward prediction rows (the same 10,069 validation homes in each track), December 2025-May 2026. The code uses Member 2's updated features and Member 3's tuned settings.

## Suggested slide: strong improvement, uneven errors

| MdAPE | Starter baseline | Final tuned model |
|---|---:|---:|
| Off-market | 20.84% | **7.79%** |
| List-aware | 7.26% | **4.23%** |

MAPE improved from 27.43% to **10.35%** off-market and from 8.09% to **5.36%** list-aware. These are validation results, not hidden-test scores. The off-market track does not use list price; list-aware does, so the tracks answer different questions.

## Suggested slide: what remains difficult

- **Detached homes:** Off-market MdAPE is **9.46%** across 3,123 homes, compared with **6.85%** for 4,843 condo apartments. List-aware: **5.45%** detached versus **3.43%** condo apartments. There are only 48 co-op apartments, too few for a reliable ranking.
- **Price extremes:** In the off-market track, the lowest actual sale-price decile has a median signed error of **+7.14%**, while the highest is **-7.15%**. Signed error is `(predicted - sold) / sold`; positive means overprediction. The deciles use actual sold price after the fact, so this is a diagnostic pattern, not a rule for new homes or proof of why it happens.
- **Tail risk:** Off-market's 90th-percentile absolute percentage error is **21.60%**; list-aware's is **11.39%**. About **0.71%** of off-market homes have more than 50% error, despite the much lower median error.

## Optional uncertainty slide

I backtested an additional, *exploratory* 90% price-interval method using residuals from earlier validation sales only. Of 10,069 homes per track, **8,045** had at least 100 prior residuals available before their listing date and evaluation month. Overall coverage was **89.77% off-market** and **88.55% list-aware**. But detached-home coverage was only **84.54%** and **80.58%**, respectively; condo-apartment coverage was **93.79%** and **96.44%**. Median interval width was **43.64%** of the prediction off-market and **21.83%** list-aware. These intervals are a separate analysis, not part of the submitted predictions. Calendar drift and changing monthly models mean this historical result is not a guaranteed 90% coverage rate for future homes.

## Voice-over draft (about 40-50 seconds)

> Our final tuned model reduced the off-market median percentage error from 20.84 to 7.79 percent. List-aware error fell from 7.26 to 4.23 percent. Both numbers come from six months of walk-forward validation.
>
> Accuracy still varies by home. For off-market valuation, detached homes have 9.46 percent median error, compared with 6.85 percent for condo apartments. At the price extremes, the model still tends to overpredict the cheapest homes and underpredict the most expensive.
>
> We also tested historical 90 percent price intervals. Overall coverage was close to 90 percent, but detached-home coverage was lower. So our conclusion is that the model has improved substantially, yet an individual estimate should be treated more cautiously for detached and unusual homes. A later-period test of property-specific uncertainty is our next step.

Visual suggestion: use `figures/01_error_overview.png` for property type and price bias. For the optional interval slide, use `figures/03_prediction_intervals.png`. Avoid showing the old baseline charts or claiming hidden-test results.
