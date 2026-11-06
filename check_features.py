"""Safety and integration checks. Run next to official data.py and models.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

import data
import features
import models


def main():
    train = data.load("listings")
    sample = train.iloc[[19, 4, 200, 93, 7]].copy()
    checks = []
    for track, transform in [("off_market", features.off_market_features),
                             ("list_aware", features.list_aware_features)]:
        safe = data.hide_answers(sample, track)
        expected = transform(safe)
        poisoned = sample.copy()
        for col in data.AFTER_SALE_COLUMNS + (["list_price"] if track == "off_market" else []):
            poisoned[col] = "DO NOT ACCESS: forbidden information"
        assert_frame_equal(expected, transform(poisoned))
        assert expected.index.equals(safe.index)
        # Categorical schema and every feature must be independent of batch order
        # or what other homes happen to be in the prediction batch.
        batched = pd.concat([transform(safe.iloc[i:i+1]) for i in range(len(safe))])
        assert_frame_equal(expected, batched)
        assert_frame_equal(expected.iloc[::-1], transform(safe.iloc[::-1]))
        unknown = safe.copy()
        for col in features.CATEGORY_LEVELS:
            unknown[col] = "UNSEEN_NEIGHBOURHOOD_OR_STYLE"
        actual = transform(unknown)
        assert actual[list(features.CATEGORY_LEVELS)].isna().all().all()
        unknown["house_area"] = [None, "0", "1,200-1,399", "6000+", "<700"]
        unknown["remarks"] = None
        transformed = transform(unknown)
        assert not np.isinf(transformed.select_dtypes("number").to_numpy()).any()
        # Fit the supplied, unchanged model and infer on the actual test schema.
        with threadpool_limits(limits=4):
            model = models.make_model(track).fit(train)
            homes = data.load(track)
            prediction = model.predict(homes)
        assert prediction.shape == (len(homes),)
        assert np.isfinite(prediction).all() and (prediction > 0).all()
        checks.append({"track": track, "n_test_rows": len(homes),
                       "feature_count": expected.shape[1],
                       "forbidden_column_poisoning": "passed",
                       "row_order_and_batch_invariance": "passed",
                       "unknown_categories": "passed", "missing_and_zero_area": "passed",
                       "real_test_inference": "passed"})
    Path("results").mkdir(exist_ok=True)
    Path("results/safety_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
