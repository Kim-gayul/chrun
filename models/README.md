# 팀 실험 코드

공개용 메인 LightGBM 파이프라인은 루트의 `pipeline.py`입니다. 이 폴더에는 원 팀 프로젝트의 나머지 네 모델 실험 코드를 보존했습니다. 데이터 경로와 딥러닝 저장 경로·장치 처리를 정리했지만 전체 장시간 학습 실험을 재검증한 코드는 아닙니다.

프로젝트 루트에서 실행합니다.

```powershell
python -m pip install -r requirements-experiments.txt
python models/RandomForest/modeling.py
python models/RandomForest/eval.py
python models/GradientBoosting/GradientBoosting_main.py
python models/XGBoost/XGBoost_main.py
python models/DeepLearning/training.py
python models/DeepLearning/final_test.py
```

Random Forest와 Deep Learning 평가에는 먼저 학습을 실행해 생성한 로컬 모델이 필요합니다. 외부에서 받은 pickle/PyTorch 모델 파일을 불러오지 마세요.

모델별 전처리·분할·탐색 조건이 다릅니다. 과거 보고서 수치를 동일 조건의 모델 순위로 해석하지 않으며, 추가 모델 비교를 진행하려면 공통 데이터 분할과 검증 방식으로 통합해야 합니다. 생성된 모델·분할 데이터는 Git에 포함하지 않습니다.
