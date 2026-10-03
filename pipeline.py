"""Reproducible LightGBM demo with separate train/validation/test partitions."""
from pathlib import Path
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_customer_churn_100k.csv"
CATEGORIES = {
    "Gender": ["Female", "Male", "Other"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
}
FEATURES = ["Age", "Gender", "Tenure", "MonthlyCharges", "Contract", "PaymentMethod", "TotalCharges"]
NUMERIC = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]
DEFAULT_PARAMS = {"learning_rate": 0.05, "num_leaves": 15, "max_depth": 6}


def prepare_features(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {', '.join(sorted(missing))}")
    result = frame[FEATURES].copy()
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column}: 결측치 또는 무한대는 허용하지 않습니다.")
    result["TotalCharges"] = result["TotalCharges"].clip(lower=0)
    for column, choices in CATEGORIES.items():
        if not result[column].isin(choices).all():
            raise ValueError(f"{column}: 지원하지 않는 값 또는 결측치가 있습니다.")
        result[column] = pd.Categorical(result[column], categories=choices)
    return result


def load_data(path=DATA_PATH):
    frame = pd.read_csv(path)
    if "Churn" not in frame:
        raise ValueError("필수 열 누락: Churn")
    target = frame["Churn"].map({"No": 0, "Yes": 1})
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Churn은 No/Yes 두 클래스를 모두 포함해야 합니다.")
    return prepare_features(frame), target.astype(int)


def split_data(features, target):
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(params=None, n_estimators=500):
    return lgb.LGBMClassifier(
        **(params or DEFAULT_PARAMS), n_estimators=n_estimators,
        random_state=42, class_weight="balanced", n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True, metric="auc",
    )


def tune_model(x_train, x_val, y_train, y_val):
    """Choose parameters using validation data only; never receive the test set."""
    rows = []
    if len(x_train) > 20000:
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=20000, stratify=y_train, random_state=42
        )
    for lr, leaves, depth in product([0.01, 0.05, 0.1], [15, 31], [4, 6]):
        if leaves > 2 ** depth:
            continue
        params = dict(learning_rate=lr, num_leaves=leaves, max_depth=depth)
        model = build_model(params, n_estimators=100)
        model.fit(x_train, y_train)
        rows.append({**params, "Validation ROC-AUC": roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])})
    results = pd.DataFrame(rows).sort_values("Validation ROC-AUC", ascending=False).reset_index(drop=True)
    best = results.iloc[0]
    return {"learning_rate": float(best.learning_rate), "num_leaves": int(best.num_leaves), "max_depth": int(best.max_depth)}, results


def train_model(x_train, x_val, y_train, y_val, params=None):
    model = build_model(params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(20, first_metric_only=True, verbose=False)])
    return model


def evaluate_model(model, x_test, y_test):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    x, y = load_data(args.data)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)
    params = DEFAULT_PARAMS
    if args.tune:
        params, results = tune_model(x_train, x_val, y_train, y_val)
        print(results.to_string(index=False))
    model = train_model(x_train, x_val, y_train, y_val, params)
    print(pd.Series(evaluate_model(model, x_test, y_test)).to_string())
