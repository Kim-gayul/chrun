import os
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# save_dir 정의
base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "saved_models")

# 모델 불러오기
model  = joblib.load(os.path.join(save_dir, "randomforest_model.pkl"))
x_test = joblib.load(os.path.join(save_dir, "x_test.pkl"))
y_test = joblib.load(os.path.join(save_dir, "y_test.pkl"))





# 예측
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# 평가
print(classification_report(y_test, y_pred))
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")





# print(grid_search.best_params_)  # 최적 파라미터
# print(grid_search.best_score_)   # 최적 점


# y_pred = grid_search.best_estimator_.predict(x_test)
# print(classification_report(y_test, y_pred))

# y_pred_proba = grid_search.best_estimator_.predict_proba(x_test)[:, 1]
# auc_score = roc_auc_score(y_test, y_pred_proba)
# print(f"ROC-AUC Score: {auc_score:.4f}")


