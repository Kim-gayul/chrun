from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from data_scaling import *
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

path = Path(__file__).resolve().parents[2] / 'data' / 'synthetic_customer_churn_100k.csv'
data = load_data(path)
x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(data)
model = RandomForestClassifier(random_state=0)
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    
    
}

grid_search = RandomizedSearchCV(
    estimator=model,      # 모델
    param_distributions=params, # 파라미터 그리드
    cv=5,                 # 교차검증 횟수
    scoring='accuracy',   # 평가 지표
    n_jobs=-1,
    random_state=0,
    n_iter=20                                    # 전체 CPU 사용
)
print("학습시작")

grid_search.fit(x_train, y_train)

import os
import joblib

# 현재 파일 기준 saved_models 폴더 (같은 레벨)
base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)

# 모델 저장
joblib.dump(grid_search.best_estimator_, os.path.join(save_dir, "randomforest_model.pkl"))

# 전처리 변수 저장
joblib.dump(x_train, os.path.join(save_dir, "x_train.pkl"))
joblib.dump(x_val,   os.path.join(save_dir, "x_val.pkl"))
joblib.dump(x_test,  os.path.join(save_dir, "x_test.pkl"))
joblib.dump(y_train, os.path.join(save_dir, "y_train.pkl"))
joblib.dump(y_val,   os.path.join(save_dir, "y_val.pkl"))
joblib.dump(y_test,  os.path.join(save_dir, "y_test.pkl"))

print(f"저장 완료: {save_dir}")
