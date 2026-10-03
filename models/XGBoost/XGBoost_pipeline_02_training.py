# XGBoost_pipeline_02_training.py
# -*- coding: utf-8 -*-
"""
[Pipeline 2] XGBoost 모델 학습 및 평가
-------------------------------------------------------------------------
이 파일이 하는 일:
  pipeline_01에서 전처리된 데이터를 받아서
  XGBoost 모델을 학습시키고 성능을 평가합니다.

  출력 결과:
    - 모델 평가 지표 테이블 (Accuracy, Precision, Recall, F1, ROC-AUC)
    - 분류 리포트 (클래스별 상세 성능)
    - 피처 중요도 그래프 (어떤 변수가 이탈 예측에 중요한지)
    - 모델 저장 (xgb_model.pkl / xgb_processor.pkl)
"""

import joblib      # 모델 저장/불러오기 도구
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from XGBoost_pipeline_01_preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# 저장할 파일 경로 (변경이 필요하면 여기서만 수정)
MODEL_PATH     = 'models/XGBoost/saved_model/xgb_model.pkl'      # 학습 완료된 XGBoost 모델
PROCESSOR_PATH = 'models/XGBoost/saved_model/xgb_processor.pkl'  # 전처리 객체 (app.py에서 새 고객 입력값 변환 시 사용)


def run_model_training_and_evaluation(X_train, X_test, y_train, y_test, processor, best_params=None):
    """
    XGBoost 모델 학습 및 평가 함수
    최적화된 파라미터로 모델을 학습하고 성능을 리포팅/저장하는 함수

    """

    if X_train is None:
        print("[오류] 입력 데이터가 올바르지 않습니다.")
        return None

    
    # 모델 하이퍼파라미터 설정 (튜닝된 값이 없으면 기본값 사용)
    n_estimators     = best_params.get('n_estimators',     300) if best_params else 300
    max_depth        = best_params.get('max_depth',          6) if best_params else 6
    learning_rate    = best_params.get('learning_rate',    0.1) if best_params else 0.1
    subsample        = best_params.get('subsample',        0.8) if best_params else 0.8
    colsample_bytree = best_params.get('colsample_bytree', 0.8) if best_params else 0.8

    print(f"최종 모델 파라미터 -> n_estimators: {n_estimators}, max_depth: {max_depth}, "
          f"learning_rate: {learning_rate}, subsample: {subsample}, colsample_bytree: {colsample_bytree}")

    # 클래스 불균형 보정
    # 이탈 고객이 적어도 모델이 이탈 패턴을 충분히 학습하도록 가중치 부여
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    # XGBoost 모델 생성 및 학습
    model = XGBClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        scale_pos_weight = scale_pos,
        eval_metric      = 'logloss',
        random_state     = 42,
        n_jobs           = -1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    print("모델 학습 완료")

   
    # 모델 성능 평가
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 65)
    print("            📊 XGBoost 모델 평가 결과")
    print("=" * 65)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}  ← 전체 예측 중 맞춘 비율")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}  ← 이탈 예측 중 실제 이탈 비율")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}  ← 실제 이탈 고객을 얼마나 잡았는지")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}  ← Precision과 Recall의 균형 점수")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}  ← 불균형 데이터 핵심 지표 (1에 가까울수록 좋음)")
    print("=" * 65)
    print(f"  * 검증 데이터: {len(y_test):,}건  |  실제 이탈률: {y_test.mean():.1%}")
    print("=" * 65)

    print("\n분류 리포트 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Not Churned (0)', 'Churned (1)']))

    
    # 6. 피처 중요도 그래프 저장
    ohe_features = (
        processor.named_transformers_['category']
        .named_steps['ohe']
        .get_feature_names_out(CATEGORICAL_COLUMNS)
        .tolist()
    )
    all_features = ohe_features + NUMERIC_COLUMNS

    importance_df = pd.DataFrame({
        'Feature'   : all_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    # 그래프 생성 및 이미지 파일로 저장
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance for Churn Prediction', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('models/XGBoost/xgb_feature_importance.png', dpi=150)
    plt.close()
    print("피처 중요도 그래프 저장 완료: 'xgb_feature_importance.png'")

   
    # 모델 및 전처리 객체 저장 (.pkl)
    
    # 모델 저장
    # 학습 완료된 모델을 파일로 저장해두면
    # 다음 실행 때 다시 학습할 필요 없이 바로 불러와서 예측 가능
    # app.py 사용자 모드에서 이 파일을 불러와 이탈 확률 예측에 사용
    from pathlib import Path
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"모델 저장 완료: '{MODEL_PATH}'")

    # 전처리 객체 저장
    joblib.dump(processor, PROCESSOR_PATH)
    print(f"전처리 객체 저장 완료: '{PROCESSOR_PATH}'")

    return model


if __name__ == '__main__':
    try:
        import XGBoost_pipeline_01_preprocessing as p1
        X_train, X_test, y_train, y_test, processor = p1.run_preprocessing()
        if X_train is not None:
            run_model_training_and_evaluation(X_train, X_test, y_train, y_test, processor)
    except ImportError:
        print("[오류] 'XGBoost_pipeline_01_preprocessing.py' 파일이 동일 폴더에 있어야 합니다.")
