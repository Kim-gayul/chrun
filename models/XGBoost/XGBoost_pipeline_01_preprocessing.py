# XGBoost_pipeline_01_preprocessing.py
# -*- coding: utf-8 -*-
"""
[Pipeline 1] 데이터 전처리
-------------------------------------------------------------------------
1. 데이터 로드
2. 전처리 파이프라인 구성 (OHE + StandardScaler)
3. Train / Test 분할
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


# 데이셋의 컬럼명을 지정
# 컬럼명이 바뀌면 여기서만 수정!
CATEGORICAL_COLUMNS = ['Gender', 'Contract', 'PaymentMethod'] #문자 컬럼 
NUMERIC_COLUMNS     = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges'] #숫자 컬럼
DROP_COLUMNS        = ['CustomerID'] # 학습제외 컬럼
TARGET_COLUMN       = 'Churn'
DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'synthetic_customer_churn_100k.csv'


#1. 전처리 파이프라인 구성
# 범주형/수치형 데이터를 각각 다르게 처리하는 전처리 파이프라인 생성
def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler())
    ])
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("category", category_pipeline, CATEGORICAL_COLUMNS),
        ("number",   numeric_pipeline,  NUMERIC_COLUMNS)
    ])
    return preprocessor


def run_preprocessing():
    if not os.path.exists(DATA_PATH):
        print(f"[오류] '{DATA_PATH}' 파일이 현재 디렉토리에 없음.")
        return None, None, None, None, None

    df = pd.read_csv(DATA_PATH)
    print(f"데이터 크기: {df.shape}")
    print(f"결측치:\n{df.isnull().sum()}")

    # 타겟 인코딩
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COLUMN])
    print(f"타겟 변수(Churn) 변환: 'Yes' -> 1, 'No' -> 0  |  Churn 비율: {y.mean():.3f}")

    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    
    # 2. 데이터셋 분할 로직
    # stratify: train/test 내 Churn 비율 동일하게 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train 기준 fit → test는 transform만 (데이터 누수 방지)
    processor = build_preprocessor()
    X_train_processed = processor.fit_transform(X_train)
    X_test_processed  = processor.transform(X_test)

    print(f"Train: {X_train_processed.shape}  |  Test: {X_test_processed.shape}")
    print(f"Train Churn 비율: {y_train.mean():.3f}  |  Test Churn 비율: {y_test.mean():.3f}")

    return X_train_processed, X_test_processed, y_train, y_test, processor


if __name__ == '__main__':
    run_preprocessing()
