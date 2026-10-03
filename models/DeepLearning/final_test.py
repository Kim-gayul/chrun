import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from deep_learning_model import *
from data_scaling import *

path = Path(__file__).resolve().parents[2] / 'data' / 'synthetic_customer_churn_100k.csv'
model_path = Path(__file__).resolve().parent / 'saved_model' / 'deep_model.pt'
train_loader, val_loader, test_loader = data_loaders(path, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.6
model = torch.load(model_path, map_location=device, weights_only=False).to(device)


# 테스트 진행
print("테스트 시작")
model.eval()

final_preds = []
final_targets = []
final_probs = []  

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        
        # 확률 값 계산
        pred_prob = torch.sigmoid(outputs)
        # 추론 결과 계산
        predicted = (pred_prob >= threshold).float()

        # numpy배열은 cpu만 사용가능
        final_preds.extend(predicted.cpu().numpy().flatten())
        final_targets.extend(batch_y.cpu().numpy().flatten())
        final_probs.extend(pred_prob.cpu().numpy().flatten())


# 5. 최종 평가 지표 출력
print("[오차 행렬 (Confusion Matrix)]")
print(confusion_matrix(final_targets, final_preds))

print("[상세 분류 보고서 (Classification Report)]")
print(classification_report(final_targets, final_preds, target_names=['Stay (0)', 'Churn (1)']))

auc = roc_auc_score(final_targets, final_probs)
print(f"[ROC-AUC Score]: {auc:.4f}")
