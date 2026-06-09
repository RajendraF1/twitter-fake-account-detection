from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.prediction import load_artifacts, predict_new_data


st.set_page_config(
    page_title="Fake Account Screening Tool",
    page_icon="🛡️",
    layout="wide"
)


REQUIRED_COLUMNS = [
    "followers_count",
    "friends_count",
    "post_count",
    "location",
    "lang",
    "description",
    "created_at"
]


@st.cache_resource
def get_artifacts():
    """
    Load model artifacts once and cache them.
    """
    model, scaler, lang_mapping = load_artifacts(
        model_path="models/slp_model.pkl",
        scaler_path="models/scaler.pkl",
        lang_mapping_path="models/lang_mapping.pkl"
    )
    return model, scaler, lang_mapping


def validate_input_columns(df: pd.DataFrame):
    """
    Validate whether uploaded dataframe contains required columns.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing_cols


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load uploaded CSV or Excel file into a DataFrame.
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")


def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to downloadable CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def build_single_input_dataframe(
    followers_count,
    friends_count,
    post_count,
    location,
    lang,
    description,
    created_at
) -> pd.DataFrame:
    """
    Build a single-row DataFrame from manual input form.
    """
    return pd.DataFrame([
        {
            "followers_count": followers_count,
            "friends_count": friends_count,
            "post_count": post_count,
            "location": location,
            "lang": lang,
            "description": description,
            "created_at": created_at
        }
    ])


def render_header():
    st.title("Fake Account Screening Tool")
    st.markdown(
        """
        This web app allows users to test a trained machine learning model for fake account detection.

        Available modes:
        - **Single Prediction** for testing one account manually
        - **Batch Prediction** for analyzing multiple accounts from a CSV/XLSX file
        """
    )


def render_sidebar():
    st.sidebar.header("About")
    st.sidebar.write(
        """
        **Model**: Single Layer Perceptron (SLP)  
        **Pipeline**: preprocessing → scaling → prediction  
        **Output**: label, probability, risk level
        """
    )

    st.sidebar.header("Required Input Fields")
    st.sidebar.code("\n".join(REQUIRED_COLUMNS))


def render_single_prediction(model, scaler, lang_mapping):
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        followers_count = st.number_input("Followers Count", min_value=0, value=100)
        friends_count = st.number_input("Friends Count", min_value=0, value=100)
        post_count = st.number_input("Post Count", min_value=0, value=10)
        lang = st.text_input("Language", value="en")

    with col2:
        location = st.text_input("Location", value="")
        created_at = st.date_input("Created At")
        description = st.text_area("Description", value="")

    if st.button("Predict Single Account", use_container_width=True):
        try:
            df_single = build_single_input_dataframe(
                followers_count=followers_count,
                friends_count=friends_count,
                post_count=post_count,
                location=location,
                lang=lang,
                description=description,
                created_at=str(created_at)
            )

            result_df = predict_new_data(
                df_new=df_single,
                model=model,
                scaler=scaler,
                lang_mapping=lang_mapping
            )

            result = result_df.iloc[0]

            st.success("Prediction completed successfully.")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Prediction", result["prediction"])
            metric_col2.metric("Fake Probability", f"{result['fake_probability']:.4f}")
            metric_col3.metric("Risk Level", result["risk_level"])

            st.markdown("### Prediction Detail")
            st.dataframe(result_df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during single prediction: {e}")


def render_batch_prediction(model, scaler, lang_mapping):
    st.subheader("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload a CSV or XLSX file",
        type=["csv", "xlsx"],
        key="batch_uploader"
    )

    st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))

    if uploaded_file is not None:
        try:
            df_uploaded = load_uploaded_file(uploaded_file)

            st.markdown("### Uploaded Data Preview")
            st.dataframe(df_uploaded.head(), use_container_width=True)

            missing_cols = validate_input_columns(df_uploaded)
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return

            if st.button("Run Batch Prediction", use_container_width=True):
                result_df = predict_new_data(
                    df_new=df_uploaded,
                    model=model,
                    scaler=scaler,
                    lang_mapping=lang_mapping
                )

                total_accounts = len(result_df)
                fake_count = (result_df["prediction"] == "Fake").sum()
                high_risk_count = (result_df["risk_level"] == "High").sum()
                avg_fake_probability = result_df["fake_probability"].mean()

                st.success("Batch prediction completed successfully.")

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Total Accounts", total_accounts)
                metric_col2.metric("Predicted Fake", int(fake_count))
                metric_col3.metric("High Risk", int(high_risk_count))
                metric_col4.metric("Avg Fake Probability", f"{avg_fake_probability:.4f}")

                st.markdown("### Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                st.markdown("### Risk Level Distribution")
                risk_counts = result_df["risk_level"].value_counts()
                st.bar_chart(risk_counts)

                csv_bytes = convert_df_to_csv_bytes(result_df)
                st.download_button(
                    label="Download Prediction Results as CSV",
                    data=csv_bytes,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"An error occurred during batch prediction: {e}")


def main():
    render_header()
    render_sidebar()

    try:
        model, scaler, lang_mapping = get_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        render_single_prediction(model, scaler, lang_mapping)

    with tab2:
        render_batch_prediction(model, scaler, lang_mapping)


if __name__ == "__main__":
    main()