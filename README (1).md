# Salary Predictor — Streamlit App

Predicts Annual Salary (LPA) for an employee, using a Gradient Boosting
Regressor trained on the cleaned/outlier-removed employee salary dataset
(R² = 0.779, MAE = ₹3.32 LPA on held-out test data — the best of 4
models tried: Linear Regression, Random Forest, Gradient Boosting, XGBoost).

## Folder contents
```
salary_app/
├── app.py                  # the Streamlit app (single form + batch upload)
├── train.py                 # training script (re-run if the dataset changes)
├── requirements.txt
└── model/
    ├── salary_model.joblib     # trained Gradient Boosting model
    ├── scaler.joblib           # StandardScaler fit on training features
    ├── feature_order.joblib    # exact column order the model expects
    └── encodings.joblib        # category -> number mappings + model metadata
```

## Run it locally
```bash
cd salary_app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy — Streamlit Community Cloud
1. Push this whole `salary_app/` folder to a GitHub repo (keep `app.py`
   and `requirements.txt` at the root, `model/` as a subfolder).
   ```bash
   cd salary_app
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   **Verify on GitHub that `model/` shows all 4 `.joblib` files before
   deploying** — a missing model folder is the #1 cause of deploy errors.
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**,
   pick this repo, set main file to `app.py`, click **Deploy**.

## App features
- **Single Employee tab**: fill in a form, get an instant salary prediction
- **Batch tab**: upload an Excel sheet (one row per employee), validates
  required columns *and* category values before predicting, returns a
  downloadable results sheet
- Blank templates downloadable from within the app for both modes

## Retraining
If you get a new/updated dataset, edit `RAW_PATH` in `train.py` and rerun
it — it replicates the exact cleaning pipeline from the original notebook
(duplicate removal, invalid-value handling, median imputation, IQR outlier
removal, encoding) and re-saves all 4 model artifacts.
