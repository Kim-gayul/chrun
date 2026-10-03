# GradientBoosting_main.py
# -*- coding: utf-8 -*-
"""
GradientBoosting 실행 파일

전처리는 한 번만 수행하고, 같은 train/test 데이터로 튜닝과 최종 학습을 이어서 실행합니다.
"""

import GradientBoosting_01_preprocessing as pipe1
import GradientBoosting_02_training as pipe2


def main():
    """
    GradientBoosting 전체 파이프라인을 실행합니다.

    Returns:
        None: 튜닝 결과 CSV, Feature Importance 이미지, 평가 결과를 생성합니다.
    """
    print("<GradientBoosting 데이터 전처리>")

    filename = '../../data/synthetic_customer_churn_100k.csv'
    X_train, X_test, y_train, y_test = pipe1.run_preprocessing(filename)

    if X_train is None:
        print("[오류] 데이터 전처리에 실패하여 파이프라인을 중단합니다.")
        return

    print("\n" + "-------------------------------------------------------------------------")
    print("<GradientBoosting 하이퍼파라미터 튜닝>")
    print("-------------------------------------------------------------------------")

    top_10_df = pipe2.run_parameter_tuning(X_train, X_test, y_train, y_test)
    if top_10_df is None or top_10_df.empty:
        print("[오류] 튜닝 결과가 없어 파이프라인을 중단합니다.")
        return

    best_params = pipe2.extract_best_params(top_10_df)
    print(f"[튜닝 완료] 최적 파라미터 조합: {best_params}")

    print("\n" + "-------------------------------------------------------------------------")
    print("<최종 GradientBoosting 모델 학습 및 평가>")
    print("-------------------------------------------------------------------------")

    pipe2.run_model_training_and_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        best_params=best_params,
    )

    print("\n[전체 GradientBoosting 파이프라인 완료]")


if __name__ == "__main__":
    main()
