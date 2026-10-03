# CHURN · Customer Retention Analytics

![CHURN 프로젝트 홍보 배너](assets/churn-banner.png)

**합성 통신사 고객 데이터로 이탈 패턴을 탐색하고, 머신러닝 모델을 비교하는 포트폴리오 프로젝트입니다.**

팀 프로젝트에서 Random Forest, Gradient Boosting, XGBoost, LightGBM, PyTorch 신경망을 비교했습니다. 공개용 버전은 LightGBM과 Streamlit을 중심으로 데이터 준비부터 예측·평가까지 실행할 수 있도록 정리했습니다.

> 배너는 AI로 제작한 컨셉 이미지이며 실제 화면 캡처가 아닙니다. 이 프로젝트는 교육·실험용이며 실제 고객 이탈 감소나 매출 개선 효과를 검증하지 않았습니다.

## 주요 기능

- 나이, 가입 기간, 계약·결제 유형 등 7개 특성으로 고객 이탈 점수 계산
- 데이터 분포, 테스트 성능, 피처 중요도를 확인하는 Streamlit 화면
- 검증 세트만 사용하는 LightGBM 파라미터 탐색 및 조기 종료
- 학습 결과 캐시로 고객 입력마다 재학습하는 비용 절감
- 팀 프로젝트의 EDA·모델 비교 보고서와 모델별 실험 코드 보존

## 빠른 시작

Windows / Python 3.13.13에서 검증했습니다. 아래 명령은 프로젝트 루트에서 실행하세요.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# 아래 데이터 준비 절차를 완료한 뒤 실행
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux에서는 `.venv\Scripts\python.exe` 대신 `.venv/bin/python`을 사용합니다.

검증 환경의 패키지 버전을 그대로 설치하려면 `requirements.txt` 대신 `requirements-lock.txt`를 사용하세요. 다른 운영체제에서의 실행은 별도 확인이 필요합니다.

### 데이터 준비

1. [Kaggle의 Telco Customer Churn Data](https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)에서 데이터셋을 내려받습니다.
2. 압축을 풀고 `synthetic_customer_churn_100k.csv`를 `data/`에 배치합니다.
3. `streamlit run app.py`로 앱을 실행합니다. 데이터가 없으면 앱에서 준비 방법을 안내합니다.

원본 CSV와 학습 모델은 이 공개본에 포함하지 않습니다. 데이터 열·허용값·이용 조건 안내는 [데이터 안내](data/README.md)를 참고하세요.

### CLI 학습과 테스트

```powershell
.venv\Scripts\python.exe pipeline.py
.venv\Scripts\python.exe pipeline.py --tune
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`pipeline.py --data "데이터파일경로"`로 다른 위치의 동일 스키마 CSV를 지정할 수 있습니다. 테스트는 자체 생성한 소규모 fixture를 사용하므로 Kaggle CSV가 필요하지 않습니다.

## 데이터와 평가 방식

원 프로젝트에서 사용한 데이터는 100,000행·9열의 합성 데이터입니다. `CustomerID`는 특성에서 제외하고 `Churn`은 `No=0`, `Yes=1`로 변환합니다.

| 구분 | 내용 |
|---|---|
| 수치형 특성 | Age, Tenure, MonthlyCharges, TotalCharges |
| 범주형 특성 | Gender, Contract, PaymentMethod |
| 전처리 | 필수 열·값 검증, TotalCharges 음수를 0으로 보정, 고정 범주 매핑 |
| 분할 | 학습 60% / 검증 20% / 테스트 20%, 계층 분할, random_state=42 |
| 학습 | LightGBM, class_weight=balanced, 검증 세트로 조기 종료 |
| 튜닝 | 학습 데이터 최대 20,000건으로 9개 조합 학습, 검증 ROC-AUC로 선택 |
| 테스트 | Accuracy, Precision, Recall, F1, ROC-AUC; 분류 임계값 0.5 |

학습·튜닝 함수는 테스트 데이터를 받지 않습니다. 화면의 성능은 현재 모델과 테스트 데이터에서 계산하며 고정된 홍보 수치를 사용하지 않습니다. 기존 팀 보고서의 수치는 서로 다른 실험 조건의 과거 기록으로, 공개용 파이프라인의 재현 성능과 구분합니다.

모델 출력은 **미보정 이탈 점수**입니다. 클래스 가중치를 사용하므로 실제 이탈 확률과 동일하게 해석하지 않습니다. 피처 중요도와 EDA는 연관성 탐색이며 인과관계를 입증하지 않습니다.

### 공개용 파이프라인 검증 결과

2026-10-03, 원 프로젝트의 로컬 CSV로 `--tune` 흐름을 검증한 결과입니다. 기본 파라미터로 시작하는 앱의 초기 지표와는 다를 수 있습니다.

| 테스트 지표 | 값 |
|---|---:|
| Accuracy | 0.7554 |
| Precision | 0.6350 |
| Recall | 0.6164 |
| F1 | 0.6255 |
| ROC-AUC | 0.8058 |

분할은 학습 60,000 / 검증 20,000 / 테스트 20,000건입니다. 선택된 파라미터는 `learning_rate=0.01`, `num_leaves=15`, `max_depth=4`, 조기 종료 후 트리 수는 76입니다. [원시 검증 결과](docs/verification.json)와 [검증 범위](docs/VALIDATION.md)를 함께 제공합니다.

## 프로젝트 구조

```text
churn/
├── app.py                       # Streamlit 예측·평가 화면
├── pipeline.py                  # 공개용 LightGBM 학습·튜닝·평가
├── requirements.txt             # 데모 실행 의존성
├── requirements-experiments.txt # 다른 모델의 실험 의존성
├── assets/                      # 홍보 배너와 이미지 생성 기록
├── data/                        # 데이터 준비 안내와 기존 EDA 그래프
├── models/                      # 다른 4개 모델의 팀 실험 코드
├── docs/                        # 기존 보고서, 출처, 공개용 수정 기록
└── tests/                       # 분할·입력 처리·평가 분리 회귀 테스트
```

## 팀 프로젝트와 개인 기여

이 저장소는 **SKN31-2nd-2Team 팀 프로젝트**를 바탕으로 정리한 개인 포트폴리오입니다. 프로젝트 전체를 개인 단독 구현으로 주장하지 않습니다.

**Kim-gayul 원 프로젝트 기여:** PM, README 작성, LightGBM 모델링.

팀의 나머지 기여는 Gradient Boosting·학습 보고서, XGBoost·Streamlit, Random Forest·전처리 보고서, 딥러닝·발표로 구성됩니다. 원 팀 저장소와 기여자 계정은 [기여 출처](docs/ATTRIBUTION.md)에 정리했습니다. 개인 프로필 이미지와 WBS 이미지는 공개 사본에서 제외했습니다.

공개용 정리 과정에서는 LightGBM 평가 분리, 경로 처리, 입력 검증, Streamlit 캐시와 성능 표시를 수정했습니다. 상세 내용은 [공개용 변경 기록](docs/PUBLIC_RELEASE.md)을 참고하세요.

## 분석 자료

- [원 프로젝트 전처리 보고서](docs/original_preprocessing_report.md)
- [원 프로젝트 모델 학습 보고서](docs/original_training_report.md)
- [추가 모델 실험 실행 안내](models/README.md)

## 한계와 개선 방향

- 합성 데이터에서 학습한 결과이며 실제 고객 데이터로 일반화 성능을 확인하지 않았습니다.
- 다른 모델의 과거 실험은 분할·튜닝 조건이 다르므로 공정한 최종 순위로 해석하지 않습니다.
- 반복적으로 테스트 점수를 참고해 모델을 선택하면 간접적인 과적합이 생깁니다. 최종 비교에는 별도의 홀드아웃 또는 중첩 교차검증이 필요합니다.
- 점수 보정, 비용 기반 임계값 선택, 시간 기반 검증, 공정성 분석은 후속 과제입니다.
- 모델 평가·튜닝 화면은 로컬 데모 기능이며 사용자 인증이나 서비스 운영 기능은 제공하지 않습니다.

## 출처와 이용 조건

데이터는 Kaggle 원 배포 페이지의 조건을 따릅니다. 이 공개본에서는 재배포 조건을 확정하지 못해 원본 CSV를 제외했습니다. 팀 코드의 라이선스를 임의로 새로 부여하지 않았으며 MIT 등 별도 라이선스는 포함하지 않았습니다. 재사용·재배포 권한은 원 기여자의 조건을 확인하세요.
