# XGBoost_pipeline_03_tuning.py
# -*- coding: utf-8 -*-
"""
[Pipeline 3] XGBoost 하이퍼파라미터 튜닝
-------------------------------------------------------------------------
- 파라미터 조합을 순회하며 ROC-AUC / Accuracy 수집
- ROC-AUC 기준 상위 10개 테이블 출력 및 CSV 저장
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 다른 파이프라인(전처리) 모듈 연결 확인
try:
    import XGBoost_pipeline_01_preprocessing as p1
except ImportError:
    print("[오류] 'XGBoost_pipeline_01_preprocessing.py' 파일이 동일한 폴더에 있어야 합니다.")
    exit()


def run_parameter_tuning():
    # 전처리 모듈에서 데이터(X_train, X_test 등)를 받아옴
    X_train, X_test, y_train, y_test, processor = p1.run_preprocessing()
    if X_train is None:
        return None
    print("-" * 65)

    # 테스트할 파라미터 후보군 정의 (그리드 서치 범위)
    #수정이 필요하면 이 리스트 내부 값을 변경
    learning_rates    = [0.01, 0.05, 0.1]
    max_depths        = [4, 6, 8]
    subsamples        = [0.7, 0.8]
    colsample_bytrees = [0.7, 0.8]

    # 10만개 데이터 전체 대신 20,000개 샘플로 빠르게 경향성 파악
    sample_size = min(20000, len(X_train))
    sample_idx  = np.random.RandomState(42).choice(len(X_train), sample_size, replace=False)
    X_sample    = X_train[sample_idx]
    y_sample    = y_train[sample_idx] if hasattr(y_train, '__getitem__') else y_train.iloc[sample_idx]

    scale_pos = (y_sample == 0).sum() / (y_sample == 1).sum()
    
    #모델 학습 및 성능 기록용 리스트
    results_list = []
    trial_count  = 1
    
    #모든 조합(4중 for문)을 돌며 모델 학습 및 성능 측정
    for lr in learning_rates:
        for depth in max_depths:
            for sub in subsamples:
                for col in colsample_bytrees:
                    print(f"[Trial {trial_count:02d}] lr: {lr}, max_depth: {depth}, "
                          f"subsample: {sub}, colsample_bytree: {col}")
                    
                    # XGBoost 모델 객체 생성
                    model = XGBClassifier(
                        n_estimators     = 100,  # 경향성 파악용으로 가볍게 제한
                        learning_rate    = lr,
                        max_depth        = depth,
                        subsample        = sub,
                        colsample_bytree = col,
                        scale_pos_weight = scale_pos,
                        eval_metric      = 'logloss',
                        random_state     = 42,
                        n_jobs           = -1,
                        verbosity        = 0
                    )
                    #학습 및 예측
                    model.fit(X_sample, y_sample)
                    y_pred       = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    results_list.append({
                        'Trial'           : trial_count,
                        'learning_rate'   : lr,
                        'max_depth'       : depth,
                        'subsample'       : sub,
                        'colsample_bytree': col,
                        'Accuracy'        : round(accuracy_score(y_test, y_pred),       4),
                        'ROC-AUC'         : round(roc_auc_score(y_test, y_pred_proba),  4)
                    })
                    trial_count += 1
    #성능 결과 분석 및 상위 10개 출력
    df_results = pd.DataFrame(results_list)
    
    # ROC-AUC 점수가 높은 순으로 정렬하여 상위 10개 추출
    top_10 = (df_results
              .sort_values(by=['ROC-AUC', 'Accuracy'], ascending=False)
              .head(10)
              .reset_index(drop=True))
    top_10.index     = top_10.index + 1
    top_10.index.name = 'Rank'

    print("-" * 65)
    print("하이퍼파라미터 튜닝 성능 기준 상위 10개 결과")
    print("-" * 65)
    print(top_10.to_string())
    print("-" * 65)

    output_csv = 'models/XGBoost/xgb_tuning_top10_results.csv'
    top_10.to_csv(output_csv, encoding='utf-8-sig')
    print(f"튜닝 결과 저장: '{output_csv}'\n")

    return top_10


if __name__ == '__main__':
    run_parameter_tuning()