import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="wide")

MODEL_DIR = Path(__file__).parent / "model"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "salary_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_order = joblib.load(MODEL_DIR / "feature_order.joblib")
    encodings = joblib.load(MODEL_DIR / "encodings.joblib")
    return model, scaler, feature_order, encodings


model, scaler, feature_order, enc = load_artifacts()

JOB_LEVELS = list(enc['job_mapping'].keys())
EDUCATION_LEVELS = list(enc['education_mapping'].keys())
DEPARTMENTS = list(enc['department_mapping'].keys())
CITIES = list(enc['city_mapping'].keys())
GENDERS = list(enc['gender_mapping'].keys())
REMOTE_OPTIONS = list(enc['remote_work_mapping'].keys())

RAW_COLUMNS = [  # what a human-readable input row looks like, before encoding
    'Age', 'Gender', 'Education', 'Experience_Years', 'Department', 'Job_Level',
    'Performance_Rating', 'Certifications', 'Overtime_Hours', 'Remote_Work',
    'City', 'Company_Tenure', 'Projects_Completed', 'Skill_Score'
]


def encode_row(row: dict) -> pd.DataFrame:
    """Take a dict of human-readable values and encode it to match model feature_order."""
    encoded = {
        'Age': row['Age'],
        'Gender': enc['gender_mapping'][row['Gender']],
        'Education': enc['education_mapping'][row['Education']],
        'Experience_Years': row['Experience_Years'],
        'Department': enc['department_mapping'][row['Department']],
        'Job_Level': enc['job_mapping'][row['Job_Level']],
        'Performance_Rating': row['Performance_Rating'],
        'Certifications': row['Certifications'],
        'Overtime_Hours': row['Overtime_Hours'],
        'Remote_Work': enc['remote_work_mapping'][row['Remote_Work']],
        'City': enc['city_mapping'][row['City']],
        'Company_Tenure': row['Company_Tenure'],
        'Projects_Completed': row['Projects_Completed'],
        'Skill_Score': row['Skill_Score'],
    }
    return pd.DataFrame([encoded])[feature_order]


def predict_salary(X: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


st.title("💰 Salary Predictor")
st.caption(
    f"Predicts Annual Salary (LPA) using a **{enc['best_model_name']}** model "
    f"(R² = {enc['r2_score']:.3f}, MAE = ₹{enc['mae']:.2f} LPA on test data)."
)

tab1, tab2 = st.tabs(["🧑 Single Employee", "📊 Batch (Excel Upload)"])

# =====================================================================
# TAB 1 — Single employee form
# =====================================================================
with tab1:
    st.subheader("Enter employee details")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=70, value=28)
        gender = st.selectbox("Gender", GENDERS)
        education = st.selectbox("Education", EDUCATION_LEVELS)
        experience = st.number_input("Experience (Years)", min_value=0.0, max_value=45.0, value=5.0, step=0.5)
        department = st.selectbox("Department", DEPARTMENTS)
    with c2:
        job_level = st.selectbox("Job Level", JOB_LEVELS)
        performance = st.slider("Performance Rating", 1, 5, 3)
        certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2)
        overtime = st.number_input("Overtime Hours (monthly)", min_value=0, max_value=100, value=10)
        remote = st.selectbox("Remote Work", REMOTE_OPTIONS)
    with c3:
        city = st.selectbox("City", CITIES)
        tenure = st.number_input("Company Tenure (years)", min_value=0, max_value=45, value=3)
        projects = st.number_input("Projects Completed", min_value=0, max_value=100, value=8)
        skill_score = st.slider("Skill Score", 0, 100, 70)

    if st.button("Predict Salary", type="primary"):
        row = {
            'Age': age, 'Gender': gender, 'Education': education,
            'Experience_Years': experience, 'Department': department,
            'Job_Level': job_level, 'Performance_Rating': performance,
            'Certifications': certifications, 'Overtime_Hours': overtime,
            'Remote_Work': remote, 'City': city, 'Company_Tenure': tenure,
            'Projects_Completed': projects, 'Skill_Score': skill_score,
        }
        X = encode_row(row)
        pred = predict_salary(X)[0]
        st.success(f"### Predicted Annual Salary: ₹{pred:.2f} LPA")
        st.caption(f"± ₹{enc['mae']:.2f} LPA typical error (based on test-set MAE)")

# =====================================================================
# TAB 2 — Batch Excel upload
# =====================================================================
with tab2:
    st.subheader("Upload an Excel sheet — one row per employee")

    with st.expander("📋 Required columns (click to expand)"):
        st.markdown("Your sheet must have **one row per employee** with these exact column names:")
        st.code(", ".join(RAW_COLUMNS), language=None)
        st.markdown(f"- `Gender` → one of: {', '.join(GENDERS)}")
        st.markdown(f"- `Education` → one of: {', '.join(EDUCATION_LEVELS)}")
        st.markdown(f"- `Department` → one of: {', '.join(DEPARTMENTS)}")
        st.markdown(f"- `Job_Level` → one of: {', '.join(JOB_LEVELS)}")
        st.markdown(f"- `Remote_Work` → one of: {', '.join(REMOTE_OPTIONS)}")
        st.markdown(f"- `City` → one of: {', '.join(CITIES)}")
        st.markdown("- All others (`Age`, `Experience_Years`, `Performance_Rating`, "
                     "`Certifications`, `Overtime_Hours`, `Company_Tenure`, "
                     "`Projects_Completed`, `Skill_Score`) are numeric.")

        template = pd.DataFrame(columns=(['Employee_Name'] if False else []) + RAW_COLUMNS)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            template.to_excel(writer, index=False, sheet_name="employees")
        buf.seek(0)
        st.download_button(
            "⬇️ Download blank template (.xlsx)",
            data=buf,
            file_name="salary_prediction_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx", "xls"])

    if uploaded is not None:
        try:
            df_input = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            st.stop()

        if df_input.empty:
            st.warning("The uploaded sheet has no rows.")
            st.stop()

        missing = [c for c in RAW_COLUMNS if c not in df_input.columns]
        if missing:
            st.error(f"❌ Missing {len(missing)} required column(s): `{'`, `'.join(missing)}`")
            st.stop()

        # Validate categorical values
        bad_values = {}
        for col, valid in [('Gender', GENDERS), ('Education', EDUCATION_LEVELS),
                            ('Department', DEPARTMENTS), ('Job_Level', JOB_LEVELS),
                            ('Remote_Work', REMOTE_OPTIONS), ('City', CITIES)]:
            invalid_rows = df_input[~df_input[col].isin(valid)]
            if not invalid_rows.empty:
                bad_values[col] = sorted(invalid_rows[col].unique().tolist())

        if bad_values:
            msg = "\n".join(f"- `{col}`: unrecognized value(s) {vals}" for col, vals in bad_values.items())
            st.error(f"❌ Some values don't match expected categories:\n\n{msg}")
            st.stop()

        st.success(f"✅ Loaded {len(df_input)} employee(s) — all required columns valid.")

        with st.spinner("Predicting..."):
            encoded_rows = pd.concat(
                [encode_row(r.to_dict()) for _, r in df_input.iterrows()],
                ignore_index=True
            )
            preds = predict_salary(encoded_rows)

        results = df_input.copy()
        results['Predicted_Salary_LPA'] = preds.round(2)

        st.subheader("Predictions")
        st.dataframe(results, use_container_width=True, height=min(35 * (len(results) + 1), 500))

        c1, c2 = st.columns(2)
        c1.metric("Average Predicted Salary", f"₹{results['Predicted_Salary_LPA'].mean():.2f} LPA")
        c2.metric("Range", f"₹{results['Predicted_Salary_LPA'].min():.2f} – ₹{results['Predicted_Salary_LPA'].max():.2f} LPA")

        out_buf = io.BytesIO()
        with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="predictions")
        out_buf.seek(0)
        st.download_button(
            "⬇️ Download predictions (.xlsx)",
            data=out_buf,
            file_name="salary_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Upload a file to get started, or grab the template above first.")

st.divider()
st.caption(
    f"Model: {enc['best_model_name']} trained on cleaned/outlier-removed employee salary data. "
    "Predictions are in Lakhs Per Annum (LPA)."
)
