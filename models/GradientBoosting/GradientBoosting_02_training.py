# GradientBoosting_02_training.py
# -*- coding: utf-8 -*-
"""
GradientBoosting 모델 구성, 튜닝, 학습, 평가 모듈.

전처리와 모델을 sklearn Pipeline으로 묶기 위해 ColumnTransformer와 Pipeline을 사용합니다.
"""
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


Params = dict[str, float | int]
Metrics = Optional[dict[str, list[str]]]


def build_model_pipeline(X: pd.DataFrame, params: Optional[Params] = None) -> Pipeline:
    """
    ColumnTransformer와 GradientBoostingClassifier를 하나의 Pipeline으로 만듭니다.

    Args:
        X (pd.DataFrame): 컬럼 타입을 확인할 피처 데이터.
        params (Params | None): 모델 파라미터.

    Returns:
        Pipeline: 전처리와 모델이 연결된 sklearn Pipeline.
    """
    params = params or {}
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = GradientBoostingClassifier(
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 150)),
        max_depth=int(params.get("max_depth", 3)),
        random_state=42,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def run_parameter_tuning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    같은 train/test 데이터에서 GradientBoosting 파라미터 조합을 검증합니다.

    Args:
        X_train (pd.DataFrame): 학습용 피처 데이터.
        X_test (pd.DataFrame): 검증용 피처 데이터.
        y_train (pd.Series): 학습용 정답 데이터.
        y_test (pd.Series): 검증용 정답 데이터.

    Returns:
        pd.DataFrame | None: ROC-AUC 기준 상위 10개 결과.
            입력 데이터가 없으면 None을 반환합니다.
    """
    if X_train is None:
        print("[오류] 입력 데이터가 올바르지 않습니다.")
        return None

    tuning_start_time = time.perf_counter()

    learning_rates = [0.03, 0.05, 0.1]
    n_estimators_list = [80, 120, 160]
    max_depths = [2, 3]

    X_train_sample = X_train.sample(n=min(15000, len(X_train)), random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_sample)

    results: list[dict[str, float | int]] = []
    trial_count = 1

    for learning_rate in learning_rates:
        for n_estimators in n_estimators_list:
            for max_depth in max_depths:
                trial_start_time = time.perf_counter()
                params: Params = {
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                }

                print(f"[Trial {trial_count:02d}] 검증 진행 중 -> {params}")
                model_pipeline = build_model_pipeline(X_train_sample, params)
                model_pipeline.fit(
                    X_train_sample,
                    y_train_sample,
                    model__sample_weight=sample_weight,
                )

                y_pred = model_pipeline.predict(X_test)
                y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
                elapsed_seconds = time.perf_counter() - trial_start_time

                results.append(
                    {
                        "Trial": trial_count,
                        **params,
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "ROC-AUC": round(roc_auc_score(y_test, y_pred_proba), 4),
                        "ElapsedSeconds": round(elapsed_seconds, 2),
                    }
                )
                trial_count += 1

    top_10_results = (
        pd.DataFrame(results)
        .sort_values(by=["ROC-AUC", "Accuracy"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_10_results.index = top_10_results.index + 1
    top_10_results.index.name = "Rank"

    print("-------------------------------------------------------------------------")
    print("GradientBoosting 하이퍼파라미터 튜닝 상위 10개 결과")
    print("-------------------------------------------------------------------------")
    print(top_10_results.to_string())
    print("-------------------------------------------------------------------------")

    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb_tuning_top10_results.csv")
    top_10_results.to_csv(output_csv, encoding="utf-8-sig")
    print(f"튜닝 상위 10개 결과 저장 완료: {output_csv}")
    print(f"전체 하이퍼파라미터 튜닝 소요 시간: {time.perf_counter() - tuning_start_time:.2f}초")

    return top_10_results


def extract_best_params(top_10_results: pd.DataFrame) -> Params:
    """
    튜닝 결과의 1위 행에서 최종 학습용 파라미터를 추출합니다.

    Args:
        top_10_results (pd.DataFrame): run_parameter_tuning 함수가 반환한 결과.

    Returns:
        Params: learning_rate, n_estimators, max_depth 딕셔너리.
    """
    best_row = top_10_results.iloc[0]
    return {
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": int(best_row["max_depth"]),
    }


def save_feature_importance(model_pipeline: Pipeline, output_filename: str) -> None:
    """
    학습된 Pipeline의 Feature Importance 그래프를 저장합니다.

    Args:
        model_pipeline (Pipeline): 학습이 완료된 sklearn Pipeline.
        output_filename (str): 저장할 이미지 파일명.

    Returns:
        None: 이미지 파일만 저장합니다.
    """
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model_pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df.head(20),
        palette="viridis",
        hue="Feature",
        legend=False,
    )
    plt.title("GradientBoosting Feature Importance for Churn Prediction", fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Feature Importance 그래프 저장 완료: {output_path}")


def run_model_training_and_evaluation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_params: Optional[Params] = None,
) -> Metrics:
    """
    최적 파라미터로 GradientBoosting Pipeline을 학습하고 평가합니다.

    Args:
        X_train (pd.DataFrame): 학습용 피처 데이터.
        X_test (pd.DataFrame): 검증용 피처 데이터.
        y_train (pd.Series): 학습용 정답 데이터.
        y_test (pd.Series): 검증용 정답 데이터.
        best_params (Params | None): 튜닝으로 선택된 모델 파라미터.

    Returns:
        Metrics: Train/Test 평가 지표 딕셔너리. 입력 오류가 있으면 None.
    """
    if X_train is None:
        print("[오류] 입력 데이터가 올바르지 않습니다.")
        return None

    model_pipeline = build_model_pipeline(X_train, best_params)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model_pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    print(f"GradientBoosting Pipeline 학습 완료: {best_params or '기본값'}")

    y_train_pred = model_pipeline.predict(X_train)
    y_train_proba = model_pipeline.predict_proba(X_train)[:, 1]
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\n------------------ [ GradientBoosting 모델 평가 결과 ] ------------------")
    print(f"1. 정확도(Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"2. ROC-AUC 점수    : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n3. 분류 리포트(Classification Report):")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------------------------------------")

    save_feature_importance(model_pipeline, "gb_feature_importance.png")

    return {
        "평가 지표(Metric)": ["정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "ROC-AUC"],
        "훈련 데이터 성능 (Train)": [
            f"{accuracy_score(y_train, y_train_pred):.4f}",
            f"{precision_score(y_train, y_train_pred):.4f}",
            f"{recall_score(y_train, y_train_pred):.4f}",
            f"{f1_score(y_train, y_train_pred):.4f}",
            f"{roc_auc_score(y_train, y_train_proba):.4f}",
        ],
        "검증 데이터 성능 (Test)": [
            f"{accuracy_score(y_test, y_pred):.4f}",
            f"{precision_score(y_test, y_pred):.4f}",
            f"{recall_score(y_test, y_pred):.4f}",
            f"{f1_score(y_test, y_pred):.4f}",
            f"{roc_auc_score(y_test, y_pred_proba):.4f}",
        ],
    }


if __name__ == "__main__":
    try:
        import GradientBoosting_01_preprocessing as p1

        split_data = p1.run_preprocessing()
        if split_data[0] is not None:
            run_model_training_and_evaluation(*split_data)
    except ImportError as error:
        print(f"[오류] 전처리 모듈을 불러올 수 없습니다: {error}")
