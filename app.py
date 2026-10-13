from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
)


def rmsle(actual, predicted):
    actual = np.maximum(np.asarray(actual, dtype=float), 0)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    return float(np.sqrt(np.mean(
        (np.log1p(actual) - np.log1p(predicted)) ** 2
    )))


st.title("Retail Demand Forecasting")
st.caption("Kaggle Store Sales | LightGBM | Data Science Portfolio")

results_path = REPORTS / "validation_predictions.csv"
importance_path = REPORTS / "feature_importance.csv"
submission_path = REPORTS / "submission.csv"

if not results_path.exists():
    st.error("Missing reports/validation_predictions.csv")
    st.stop()

results = pd.read_csv(results_path, parse_dates=["date"])

required = {
    "date", "family", "sales",
    "baseline_prediction", "model_prediction"
}

if not required.issubset(results.columns):
    st.error("The validation file does not contain the required columns.")
    st.stop()

actual = results["sales"].to_numpy()
baseline = results["baseline_prediction"].to_numpy()
predicted = results["model_prediction"].to_numpy()

model_score = rmsle(actual, predicted)
baseline_score = rmsle(actual, baseline)
improvement = (
    (baseline_score - model_score) / baseline_score * 100
    if baseline_score > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Validation RMSLE", f"{model_score:.5f}")
col2.metric("Baseline RMSLE", f"{baseline_score:.5f}")
col3.metric("Improvement", f"{improvement:.2f}%")
col4.metric("Validation Rows", f"{len(results):,}")

st.info("Official Kaggle Version 1 score: 0.51413")

tab1, tab2, tab3, tab4 = st.tabs([
    "Sales Forecast",
    "Product Families",
    "Feature Importance",
    "Submission",
])

with tab1:
    st.subheader("Actual vs Predicted Daily Sales")

    daily = (
        results.groupby("date", as_index=False)
        .agg(
            Actual=("sales", "sum"),
            Baseline=("baseline_prediction", "sum"),
            LightGBM=("model_prediction", "sum"),
        )
        .sort_values("date")
    )

    st.line_chart(daily.set_index("date"))

    st.subheader("Daily forecast details")
    daily["absolute_error"] = abs(
        daily["Actual"] - daily["LightGBM"]
    )
    st.dataframe(daily, use_container_width=True)

with tab2:
    st.subheader("Explore Product Families")

    families = sorted(results["family"].astype(str).unique())
    selected = st.selectbox("Choose product family", families)

    selected_data = results[
        results["family"].astype(str) == selected
    ].copy()

    family_daily = (
        selected_data.groupby("date", as_index=False)
        .agg(
            Actual=("sales", "sum"),
            LightGBM=("model_prediction", "sum"),
            Baseline=("baseline_prediction", "sum"),
        )
        .sort_values("date")
    )

    st.line_chart(family_daily.set_index("date"))

    score = rmsle(
        selected_data["sales"],
        selected_data["model_prediction"]
    )
    st.metric("Selected Family RMSLE", f"{score:.5f}")

    summary = []
    for family, group in results.groupby("family"):
        summary.append({
            "family": family,
            "RMSLE": rmsle(
                group["sales"], group["model_prediction"]
            ),
            "actual_sales": group["sales"].sum(),
            "predicted_sales": group["model_prediction"].sum(),
        })

    family_table = pd.DataFrame(summary).sort_values(
        "RMSLE", ascending=False
    )
    st.subheader("Families with highest forecasting error")
    st.dataframe(family_table.head(15), use_container_width=True)

with tab3:
    st.subheader("Feature Importance")

    if importance_path.exists():
        importance = pd.read_csv(importance_path)

        if {"feature", "importance"}.issubset(importance.columns):
            importance = importance.sort_values(
                "importance", ascending=True
            )
            st.bar_chart(
                importance.tail(20).set_index("feature")["importance"]
            )
            st.caption(
                "Importance reflects model usage, not causal effects."
            )
        else:
            st.warning("Unexpected feature-importance file format.")
    else:
        st.warning("Feature importance report not found.")

with tab4:
    st.subheader("Kaggle Submission")

    if submission_path.exists():
        submission = pd.read_csv(submission_path)
        st.write(f"Prediction rows: {len(submission):,}")
        st.dataframe(submission.head(20), use_container_width=True)

        st.download_button(
            "Download submission.csv",
            data=submission.to_csv(index=False).encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
        )
    else:
        st.warning("reports/submission.csv not found.")
