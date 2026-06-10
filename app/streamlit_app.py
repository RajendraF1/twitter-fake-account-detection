from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import re
import pandas as pd
import numpy as np
import streamlit as st
from glob import glob
from src.prediction import load_artifacts, predict_new_data
from src.config import load_config
from src.pipeline import MODEL_REGISTRY

st.set_page_config(
    page_title="Fake Account Detection Dashboard",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "followers_count", "friends_count", "post_count",
    "location", "lang", "description", "created_at",
]

config = load_config()
MODELS_DIR = Path(config["paths"]["models_dir"])
REPORTS_DIR = Path(config["paths"]["reports_dir"])
FIGURES_DIR = Path(config["paths"]["figures_dir"])


@st.cache_resource
def get_artifacts(model_name: str):
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    lang_mapping_path = MODELS_DIR / "lang_mapping.pkl"
    return load_artifacts(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        lang_mapping_path=str(lang_mapping_path),
    )


def load_metrics(model_name: str):
    path = REPORTS_DIR / f"{model_name}_metrics.txt"
    if not path.exists():
        path = REPORTS_DIR / "metrics.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def extract_metric_value(text: str, label: str, idx: int = -1):
    pattern = rf"{label}:\s*([\d.]+)"
    matches = re.findall(pattern, text)
    if matches:
        return float(matches[idx])
    return None


def load_feature_weights(model_name: str):
    path = REPORTS_DIR / f"{model_name}_feature_weights.csv"
    if not path.exists():
        path = REPORTS_DIR / "feature_weights.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.sort_values("abs_weight", ascending=True)


def load_comparison_data():
    rows = []
    for name in MODEL_REGISTRY:
        text = load_metrics(name)
        if text is None:
            continue
        train_acc = extract_metric_value(text, "Train Accuracy")
        cv_mean = extract_metric_value(text, "CV Mean Accuracy")
        cv_std = extract_metric_value(text, "CV Std Accuracy")
        test_acc = extract_metric_value(text, "Test Accuracy")
        rows.append({
            "Model": name,
            "Train Acc": train_acc,
            "CV Mean": cv_mean,
            "CV Std": cv_std,
            "Test Acc": test_acc,
        })
    return pd.DataFrame(rows)


def validate_input_columns(df: pd.DataFrame):
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported format. Use CSV or XLSX.")


def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_single_df(followers, friends, posts, location, lang, description, created_at):
    return pd.DataFrame([{
        "followers_count": followers,
        "friends_count": friends,
        "post_count": posts,
        "location": location,
        "lang": lang,
        "description": description,
        "created_at": created_at,
    }])


def render_header():
    st.title("Fake Account Detection Dashboard")
    st.markdown(
        "Multi-model dashboard for detecting fake Twitter accounts. "
        "Compare SLP, Logistic Regression, Random Forest, and XGBoost."
    )


def render_sidebar():
    st.sidebar.header("About")
    st.sidebar.write(
        "**Models**: SLP, Logistic Regression, Random Forest, XGBoost\n\n"
        "**Pipeline**: preprocess \u2192 feature engineering \u2192 scale \u2192 predict\n\n"
        "**Output**: label (Fake/Real), probability, risk level"
    )
    st.sidebar.header("Required Columns")
    st.sidebar.code("\n".join(REQUIRED_COLUMNS))


def render_tab_dashboard():
    st.subheader("Model Comparison Dashboard")
    df_cmp = load_comparison_data()
    if df_cmp.empty:
        st.warning("No trained models found. Run `python -m scripts.train --all` first.")
        return

    st.dataframe(
        df_cmp.style.highlight_max(color="#2ecc71", axis=0, subset=["Train Acc", "CV Mean", "Test Acc"]),
        use_container_width=True,
        hide_index=True,
    )

    if "CV Mean" in df_cmp.columns and "Model" in df_cmp.columns:
        st.subheader("Cross-Validation Accuracy Comparison")
        st.bar_chart(df_cmp.set_index("Model")["CV Mean"])

    st.info("Metrics are loaded from the `reports/` directory. Run training to generate fresh results.")


def render_tab_model_detail():
    st.subheader("Model Detail")
    model_name = st.selectbox("Select Model", list(MODEL_REGISTRY.keys()), key="detail_model")

    col1, col2 = st.columns(2)
    with col1:
        metrics_text = load_metrics(model_name)
        if metrics_text:
            st.text_area("Metrics", metrics_text, height=300, disabled=True)
        else:
            st.warning("No metrics found for this model.")

    with col2:
        cm_path = FIGURES_DIR / f"{model_name}_train_confusion_matrix.png"
        if not cm_path.exists():
            cm_path = FIGURES_DIR / "train_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix (Train)", use_container_width=True)

    fw = load_feature_weights(model_name)
    if fw is not None:
        st.subheader("Feature Weights")
        st.dataframe(fw, use_container_width=True, hide_index=True)
        st.subheader("Feature Importance")
        st.bar_chart(fw.set_index("feature")["abs_weight"])
    else:
        st.warning("No feature weights found for this model.")


def render_tab_single_predict():
    st.subheader("Single Prediction")
    model_name = st.selectbox("Select Model", list(MODEL_REGISTRY.keys()), key="single_model")

    col1, col2 = st.columns(2)
    with col1:
        followers = st.number_input("Followers Count", min_value=0, value=100)
        friends = st.number_input("Friends Count", min_value=0, value=100)
        posts = st.number_input("Post Count", min_value=0, value=10)
        lang = st.text_input("Language", value="en")
    with col2:
        location = st.text_input("Location", value="")
        created_at = st.date_input("Created At")
        description = st.text_area("Description", value="")

    if st.button("Predict", use_container_width=True):
        try:
            model, scaler, lang_mapping = get_artifacts(model_name)
            df_in = build_single_df(followers, friends, posts, location, lang, description, str(created_at))
            result = predict_new_data(df_in, model, scaler, lang_mapping).iloc[0]

            st.success("Prediction completed.")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Prediction", result["prediction"])
            mc2.metric("Fake Probability", f"{result['fake_probability']:.4f}")
            mc3.metric("Risk Level", result["risk_level"])
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def render_tab_batch_predict():
    st.subheader("Batch Prediction")
    model_name = st.selectbox("Select Model", list(MODEL_REGISTRY.keys()), key="batch_model")
    uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"], key="batch_file")

    if uploaded_file is not None:
        try:
            df_in = load_uploaded_file(uploaded_file)
            st.markdown("### Uploaded Data Preview")
            st.dataframe(df_in.head(), use_container_width=True)

            missing = validate_input_columns(df_in)
            if missing:
                st.error(f"Missing columns: {missing}")
                return

            if st.button("Run Batch Prediction", use_container_width=True):
                model, scaler, lang_mapping = get_artifacts(model_name)
                result_df = predict_new_data(df_in, model, scaler, lang_mapping)

                total = len(result_df)
                fake_count = int((result_df["prediction"] == "Fake").sum())
                high_risk = int((result_df["risk_level"] == "High").sum())
                avg_prob = result_df["fake_probability"].mean()

                st.success("Batch prediction completed.")
                cols = st.columns(4)
                cols[0].metric("Total", total)
                cols[1].metric("Fake", fake_count)
                cols[2].metric("High Risk", high_risk)
                cols[3].metric("Avg Probability", f"{avg_prob:.4f}")

                st.markdown("### Results")
                st.dataframe(result_df, use_container_width=True)

                st.markdown("### Risk Distribution")
                st.bar_chart(result_df["risk_level"].value_counts())

                st.download_button(
                    label="Download CSV",
                    data=convert_df_to_csv_bytes(result_df),
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


def main():
    render_header()
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001f4ca Dashboard",
        "\U0001f50d Model Detail",
        "\U0001f9ea Single Predict",
        "\U0001f4e4 Batch Predict",
    ])

    with tab1:
        render_tab_dashboard()
    with tab2:
        render_tab_model_detail()
    with tab3:
        render_tab_single_predict()
    with tab4:
        render_tab_batch_predict()


if __name__ == "__main__":
    main()
