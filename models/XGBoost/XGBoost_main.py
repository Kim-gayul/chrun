# XGBoost_main.py
# -*- coding: utf-8 -*-
"""
[필수 라이브러리 설치]
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

-------------------------------------------------------------------------
고객 이탈 예측 XGBoost 메인 실행 파일
-------------------------------------------------------------------------
실행 순서:
  1단계 (Pipeline 01): CSV 파일 로드 → 전처리 → Train/Test 분할
  2단계 (Pipeline 03): 여러 파라미터 조합 테스트 → 최적 파라미터 선택
  3단계 (Pipeline 02): 최적 파라미터로 최종 모델 학습 → 성능 평가

실행 방법:
  터미널에서 → python XGBoost_main.py

생성되는 파일:
  xgb_tuning_top10_results.csv  : 튜닝 상위 10개 결과
  xgb_feature_importance.png    : 피처 중요도 그래프
  xgb_model.pkl                 : 저장된 학습 모델 (app.py에서 재사용)
  xgb_processor.pkl             : 저장된 전처리 객체 (app.py에서 재사용)
"""

import XGBoost_pipeline_01_preprocessing as pipe1
import XGBoost_pipeline_02_training      as pipe2
import XGBoost_pipeline_03_tuning        as pipe3


def main():

   
    # 1단계: 데이터 전처리
    X_train, X_test, y_train, y_test, processor = pipe1.run_preprocessing()

    if X_train is None:
        print("[오류] 데이터 전처리 실패. 파이프라인을 중단합니다.")
        return

   
    # 2단계: 하이퍼파라미터 튜닝
    print("\n" + "-" * 65)
    print("<2단계: 하이퍼파라미터 튜닝 진행>")
    print("-" * 65)
    top_10_df = pipe3.run_parameter_tuning()

    best_params = {
        'n_estimators'    : 300,
        'learning_rate'   : float(top_10_df.iloc[0]['learning_rate']),
        'max_depth'       : int(top_10_df.iloc[0]['max_depth']),
        'subsample'       : float(top_10_df.iloc[0]['subsample']),
        'colsample_bytree': float(top_10_df.iloc[0]['colsample_bytree']),
    }
    print(f"[튜닝 완료] 최적 파라미터: {best_params}")

    
    # 3단계: 최종 모델 학습 및 평가
    print("\n" + "-" * 65)
    print("<3단계: 최적 파라미터 기반 최종 모델 학습>")
    print("-" * 65)
    pipe2.run_model_training_and_evaluation(
        X_train, X_test, y_train, y_test,
        processor,
        best_params=best_params
    )

    print("\n" + "=" * 65)
    print("[완료] 전체 XGBoost 파이프라인이 성공적으로 완료되었습니다.")
    print("=" * 65)
    print("생성된 파일 목록:")
    print("  - xgb_tuning_top10_results.csv  : 튜닝 결과 상위 10개")
    print("  - xgb_feature_importance.png    : 피처 중요도 그래프")
    print("  - xgb_model.pkl                 : 저장된 학습 모델")
    print("  - xgb_processor.pkl             : 저장된 전처리 객체")
    print("=" * 65)


if __name__ == '__main__':
    main()