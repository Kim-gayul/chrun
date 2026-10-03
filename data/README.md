# 데이터 준비

원 프로젝트는 [Telco Customer Churn Data](https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)의 `synthetic_customer_churn_100k.csv`를 사용합니다.

원 배포 페이지에서 이용 조건을 확인하고 CSV를 다운로드한 뒤 이 폴더에 배치하세요. 원본 CSV는 공개 사본에 포함하지 않으며 `.gitignore`로 제외합니다. `eda_plots/`의 그래프는 원 팀 프로젝트에서 생성한 분석 산출물입니다.

| 열 | 값 / 처리 |
|---|---|
| CustomerID | 원본 식별 열, 모델에는 사용하지 않음 |
| Age | 수치 |
| Gender | Female, Male, Other |
| Tenure | 가입 개월 수 |
| MonthlyCharges | 월 청구 금액 |
| Contract | Month-to-month, One year, Two year |
| PaymentMethod | Bank transfer, Credit card, Electronic check, Mailed check |
| TotalCharges | 총 청구 금액, 음수는 0으로 처리 |
| Churn | No 또는 Yes, 학습 시 두 클래스 모두 필요 |

공개용 파이프라인은 결측치·무한대·알 수 없는 범주·잘못된 타깃 값을 오류로 안내합니다. 이 파일명만으로 실제 개인정보의 부재를 보장하는 것은 아니므로 다른 데이터로 교체할 경우 별도로 검토하세요.
