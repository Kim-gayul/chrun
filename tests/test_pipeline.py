import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import prepare_features, split_data, train_model, evaluate_model, tune_model


def fixture():
    rng = np.random.default_rng(42)
    n = 400
    frame = pd.DataFrame({
        "Age": rng.integers(18, 81, n), "Gender": ["Female", "Male"] * (n // 2),
        "Tenure": rng.integers(1, 73, n), "MonthlyCharges": rng.uniform(10, 150, n),
        "Contract": ["Month-to-month", "Two year"] * (n // 2),
        "PaymentMethod": ["Bank transfer"] * n, "TotalCharges": rng.uniform(-10, 10000, n),
    })
    return prepare_features(frame), pd.Series([0, 1] * (n // 2))


class PipelineTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_repeatable(self):
        x, y = fixture()
        first = split_data(x, y)
        second = split_data(x, y)
        parts = [set(part.index) for part in first[:3]]
        self.assertEqual([len(p) for p in parts], [240, 80, 80])
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        self.assertEqual(set.union(*parts), set(x.index))
        for a, b in zip(first, second):
            self.assertTrue(a.equals(b))
        for labels in first[3:]:
            self.assertAlmostEqual(labels.mean(), 0.5)

    def test_prediction_schema_and_invalid_inputs(self):
        x, _ = fixture()
        row = x.iloc[[0]].copy()
        row["TotalCharges"] = -1
        result = prepare_features(row)
        self.assertEqual(result["TotalCharges"].iloc[0], 0)
        self.assertEqual(list(result["Contract"].cat.categories), list(x["Contract"].cat.categories))
        row["Gender"] = "unsupported"
        with self.assertRaises(ValueError):
            prepare_features(row)
        with self.assertRaises(ValueError):
            prepare_features(x.drop(columns="Age"))
        row = x.iloc[[0]].copy()
        row["Age"] = np.inf
        with self.assertRaises(ValueError):
            prepare_features(row)

    def test_training_early_stopping_receives_validation_only(self):
        x, y = fixture()
        xt, xv, xs, yt, yv, ys = split_data(x, y)
        with patch("pipeline.build_model") as factory:
            train_model(xt, xv, yt, yv)
            args, kwargs = factory.return_value.fit.call_args
            self.assertIs(args[0], xt)
            self.assertIs(kwargs["eval_set"][0][0], xv)
            self.assertIs(kwargs["eval_set"][0][1], yv)
            self.assertFalse(set(xs.index) & set(xv.index))

    def test_training_tuning_and_single_customer_prediction(self):
        x, y = fixture()
        xt, xv, xs, yt, yv, ys = split_data(x, y)
        params, trials = tune_model(xt, xv, yt, yv)
        self.assertEqual(len(trials), 9)
        model = train_model(xt, xv, yt, yv, params)
        metrics = evaluate_model(model, xs, ys)
        self.assertEqual(set(metrics), {"Accuracy", "Precision", "Recall", "F1", "ROC-AUC"})
        self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))
        score = model.predict_proba(prepare_features(xs.iloc[[0]]))[0, 1]
        self.assertTrue(0 <= score <= 1)


if __name__ == "__main__":
    unittest.main()
