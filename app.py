"""Run with: streamlit run app.py"""
import pandas as pd
import streamlit as st

from pipeline import (
    ROOT, DATA_PATH, CATEGORIES, DEFAULT_PARAMS, load_data, split_data,
    train_model, tune_model, evaluate_model, prepare_features,
)

st.set_page_config(page_title="CHURN | Customer Retention Analytics", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp { background: #0b1220; }
.block-container { max-width: 1150px; padding-top: 2rem; }
h1, h2, h3 { letter-spacing: -.035em; }
div[data-testid="stMetric"] { background: #132238; border: 1px solid #25405c;
    padding: 18px; border-radius: 14px; }
</style>""", unsafe_allow_html=True)
st.caption("MACHINE LEARNING PORTFOLIO · SYNTHETIC TELECOM DATA")
st.title("CHURN")
st.subheader("고객 이탈 패턴을 탐색하고, 예측을 실험합니다.")
st.caption("LightGBM 기반 교육용 데모 · 합성 데이터 · 실제 고객에 대한 효과는 검증하지 않았습니다.")


@st.cache_resource(show_spinner=False)
def get_bundle(path, modified_ns, size, learning_rate, num_leaves, max_depth):
    x, y = load_data(path)
    xt, xv, xs, yt, yv, ys = split_data(x, y)
    params = dict(learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth)
    model = train_model(xt, xv, yt, yv, params)
    return model, evaluate_model(model, xs, ys), len(x), float(y.mean())


if not DATA_PATH.is_file():
    st.info("실행에 필요한 데이터셋을 먼저 준비해 주세요.")
    st.markdown("[Kaggle 원본 데이터](https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)를 내려받아 아래 위치에 배치하세요.")
    st.code("data/synthetic_customer_churn_100k.csv")
    st.caption("데이터 CSV는 공개 저장소에 포함하지 않습니다. 자세한 내용은 data/README.md를 확인하세요.")
    st.stop()

params = st.session_state.get("params", DEFAULT_PARAMS)
try:
    metadata = DATA_PATH.stat()
    with st.spinner("모델을 준비하고 있습니다. 최초 실행 시 학습하며 이후에는 캐시를 사용합니다."):
        model, metrics, row_count, churn_rate = get_bundle(
            str(DATA_PATH), metadata.st_mtime_ns, metadata.st_size, **params
        )
except (ValueError, OSError) as exc:
    st.error(f"데이터를 확인해 주세요: {exc}")
    st.stop()

first, second, third, fourth = st.columns(4)
first.metric("데이터 행", f"{row_count:,}")
second.metric("이탈 비율", f"{churn_rate:.1%}")
third.metric("테스트 ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
fourth.metric("테스트 Recall", f"{metrics['Recall']:.3f}")
predict_tab, model_tab, about_tab = st.tabs(["고객 이탈 예측", "모델 평가 · 튜닝", "프로젝트 소개"])

with predict_tab:
    st.subheader("고객 정보 입력")
    with st.form("customer"):
        left, right = st.columns(2)
        with left:
            age = st.slider("나이", 18, 80, 40)
            gender = st.selectbox("성별", CATEGORIES["Gender"])
            tenure = st.slider("가입 기간 (개월)", 1, 72, 12)
            contract = st.selectbox("계약 유형", CATEGORIES["Contract"])
        with right:
            monthly = st.number_input("월 청구 금액 ($)", min_value=10.0, max_value=150.0, value=65.0)
            total = st.number_input("총 청구 금액 ($)", min_value=0.0, value=780.0)
            payment = st.selectbox("결제 수단", CATEGORIES["PaymentMethod"])
        submitted = st.form_submit_button("이탈 점수 계산", type="primary")
    if submitted:
        frame = pd.DataFrame([dict(Age=age, Gender=gender, Tenure=tenure,
            MonthlyCharges=monthly, TotalCharges=total, Contract=contract, PaymentMethod=payment)])
        score = float(model.predict_proba(prepare_features(frame))[0, 1])
        st.metric("모델 이탈 점수", f"{score:.1%}")
        st.progress(score)
        st.write("검토 대상" if score >= 0.5 else "기준 미만")
        st.caption("분류 임계값은 0.5입니다. 클래스 가중치를 사용한 미보정 모델 점수이므로 실제 이탈 확률로 해석하지 마세요.")

with model_tab:
    st.subheader("독립 테스트 세트 평가")
    st.caption("학습 60% / 검증 20% / 테스트 20% · 계층 분할 · random_state=42")
    st.dataframe(pd.DataFrame({"Metric": list(metrics), "Score": list(metrics.values())}), hide_index=True)
    st.write("현재 하이퍼파라미터", params)
    st.caption(f"조기 종료 후 트리 수: {model.best_iteration_}")
    st.subheader("피처 중요도")
    importance = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values(ascending=False)
    st.bar_chart(importance)
    st.caption("중요도는 트리 분할 횟수이며 인과관계를 의미하지 않습니다.")
    if st.button("검증 세트로 파라미터 튜닝", type="secondary"):
        with st.spinner("학습 데이터 일부와 검증 세트로 9개 조합을 비교합니다."):
            x, y = load_data()
            xt, xv, _, yt, yv, _ = split_data(x, y)
            best, results = tune_model(xt, xv, yt, yv)
            st.session_state["params"] = best
            st.session_state["tuning_results"] = results
        st.rerun()
    if "tuning_results" in st.session_state:
        st.dataframe(st.session_state["tuning_results"], hide_index=True)
    st.caption("테스트 결과를 보고 다시 설계를 바꾸면 테스트에 간접적으로 맞추게 됩니다. 최종 비교에는 새 홀드아웃 또는 교차검증을 사용하세요.")

with about_tab:
    st.markdown("팀 프로젝트의 모델 비교 작업을 바탕으로 실행 흐름과 평가 분리를 정리한 공개용 포트폴리오입니다.")
    st.markdown("**본인 기여:** 김가율 — PM, README 작성, LightGBM 모델링. 다른 모델은 팀 공동 작업에서 가져온 실험 코드입니다.")
    st.markdown("**공개용 정리:** 입력 검증, 총요금 음수 보정, 데이터 분할 개선, 모델 캐시, 실제 평가 지표 표시.")
    st.markdown("상세 실행법·기여 출처·실험 한계는 README와 docs/를 참고하세요.")
