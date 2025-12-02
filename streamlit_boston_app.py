import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Path to dataset
DATA_PATH = "/data/Boston.csv"
DEFAULT_MODEL_PATH = "/model/boston.joblib"

@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def suggest_type(ser):
    if pd.api.types.is_numeric_dtype(ser):
        if ser.nunique(dropna=True) > 15:
            return "continuous"
        else:
            return "categorical (numeric)"
    else:
        return "categorical"


def try_load_model(path: str):
    """Attempt to load a joblib-exported model or pipeline.
    Returns (model_object, info_message)
    """
    p = Path(path)
    if not p.exists():
        return None, f"Model file not found at {path}"
    try:
        m = joblib.load(path)
        return m, f"Loaded model from {path}"
    except Exception as e:
        return None, f"Failed to load model: {e}"


def prepare_input(df_row: pd.DataFrame, feature_cols, cat_cols, expected_columns=None):
    """Prepare user input DataFrame for model prediction.
    - df_row: single-row DataFrame with raw user values (strings for categorical)
    - feature_cols: list of features originally selected
    - cat_cols: which of those should be treated as categorical
    - expected_columns: if provided, reindex to these columns (fill missing with 0)
    """
    X_user = df_row[feature_cols].copy()
    # Convert categorical columns to strings
    for c in cat_cols:
        if c in X_user.columns:
            X_user[c] = X_user[c].astype(str)
    # One-hot encode categorical cols
    if len(cat_cols) > 0:
        X_user_enc = pd.get_dummies(X_user, columns=cat_cols, drop_first=True)
    else:
        X_user_enc = X_user.copy()

    if expected_columns is not None:
        # Make sure all expected columns exist; add missing with 0
        X_user_enc = X_user_enc.reindex(columns=expected_columns, fill_value=0)
    return X_user_enc


def infer_expected_columns_from_model(model):
    """Try several heuristics to extract expected feature column names from the loaded model/pipeline.
    Returns list of column names or None if unknown.
    """
    # sklearn estimators often have feature_names_in_
    try:
        cols = list(model.feature_names_in_)
        return cols
    except Exception:
        pass
    # if it's a pipeline or has named_steps, try to inspect final estimator
    try:
        if hasattr(model, 'named_steps'):
            # try to pull columns from the preprocessing step if present
            # many pipelines accept DataFrame directly; we can't always extract columns
            final = model
            # check if model was saved as a dict with metadata
    except Exception:
        pass
    # model might be saved along with metadata dict
    try:
        if isinstance(model, dict):
            for key in ['feature_names', 'columns', 'feature_columns', 'X_columns']:
                if key in model and isinstance(model[key], (list, tuple)):
                    return list(model[key])
    except Exception:
        pass
    return None


def main():
    st.set_page_config(page_title="Boston: Load Model & Predict", layout="wide")
    st.title("Boston dataset — load your joblib model and predict")

    st.markdown(
        """
        This app reads `/mnt/data/Boston.csv` and builds input widgets for the following columns:
        `crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat, medv`.

        Instead of training a model, you can load your pre-trained joblib model (default path: `/mnt/data/model.joblib`) or upload one.
        The app will try to prepare the user inputs to match the model's expected features and call `model.predict(...)`.
        """
    )

    df = load_data()
    cols = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat","medv"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()

    st.sidebar.header("Model input settings")
    model_path_input = st.sidebar.text_input("Model path (joblib)", value=DEFAULT_MODEL_PATH)
    st.sidebar.write("Or upload a joblib file below (this will override the path).")
    uploaded = st.sidebar.file_uploader("Upload joblib model", type=["joblib", "pkl"], accept_multiple_files=False)

    st.header("Columns summary (min / max / suggested)")
    summary_rows = []
    for c in cols:
        ser = df[c]
        dtype = ser.dtype
        nunique = ser.nunique(dropna=True)
        minv = float(ser.min()) if pd.api.types.is_numeric_dtype(ser) else "-"
        maxv = float(ser.max()) if pd.api.types.is_numeric_dtype(ser) else "-"
        suggested = suggest_type(ser)
        summary_rows.append({"column": c, "dtype": str(dtype), "n_unique": nunique, "min": minv, "max": maxv, "suggested": suggested})
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    st.header("Build input values")
    st.write("Use the widgets below to set values you'd like to predict for.")

    # Allow user to force certain numeric columns to be categorical
    st.sidebar.header("Column type overrides")
    overrides = {}
    for c in cols:
        ser = df[c]
        suggested = suggest_type(ser)
        overrides[c] = st.sidebar.checkbox(f"Treat `{c}` as categorical", value=(suggested != "continuous"), key=f"ov_{c}")

    input_values = {}
    with st.form(key='input_form'):
        for c in cols:
            if c == 'medv':
                continue
            ser = df[c]
            is_categorical = overrides.get(c, False) or (not pd.api.types.is_numeric_dtype(ser)) or (ser.nunique(dropna=True) <= 15)
            if is_categorical:
                opts = sorted(list(map(str, ser.dropna().unique())))
                if len(opts) == 0:
                    input_values[c] = None
                else:
                    input_values[c] = st.selectbox(f"{c} (categorical)", options=opts, index=0, key=f"inp_{c}")
            else:
                minv = float(ser.min())
                maxv = float(ser.max())
                meanv = float(ser.mean())
                step = (maxv - minv) / 100.0 if maxv > minv else 1.0
                input_values[c] = st.slider(f"{c} (continuous)", min_value=minv, max_value=maxv, value=meanv, step=step, key=f"inp_{c}")
        submitted = st.form_submit_button("Save input preview")

    st.write("### Current input values")
    st.json(input_values)

    st.header("Load model & Predict")
    st.write("The app will try to load a joblib model and call `predict` on your provided input.")

    model = None
    model_info = "No model loaded yet."

    # First try uploaded file
    if uploaded is not None:
        # write uploaded file to /tmp and load
        tmp_path = Path("/tmp") / uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        model, model_info = try_load_model(str(tmp_path))
    else:
        model, model_info = try_load_model(model_path_input)

    st.write(model_info)

    # Determine feature/target choices
    target = st.selectbox("Select target column (what the model predicts)", options=cols, index=cols.index('medv'))
    feature_cols = st.multiselect("Select feature columns (those used as inputs to the model)", options=[c for c in cols if c != target], default=[c for c in cols if c != target])

    cat_cols = [c for c in feature_cols if overrides.get(c, False) or (not pd.api.types.is_numeric_dtype(df[c])) or (df[c].nunique(dropna=True) <= 15)]

    if st.button("Predict with loaded model"):
        if model is None:
            st.error("No model loaded. Provide a valid joblib model path or upload one.")
        else:
            # prepare single-row DataFrame
            df_row = pd.DataFrame([input_values])
            # ensure all feature columns exist in row (they should)
            for c in feature_cols:
                if c not in df_row.columns:
                    df_row[c] = np.nan

            expected_cols = infer_expected_columns_from_model(model)
            if expected_cols is not None:
                st.info("Model exposes expected feature columns; the app will align input to those columns.")
            else:
                st.info("Model does not expose expected feature columns (or they couldn't be inferred). The app will try a best-effort alignment and may still work if the model accepts a DataFrame directly.")

            X_user_enc = prepare_input(df_row, feature_cols, cat_cols, expected_columns=expected_cols)

            # Try to predict
            try:
                # If model is a dict-like wrapper with key 'model', unwrap
                if isinstance(model, dict) and 'model' in model:
                    model_obj = model['model']
                else:
                    model_obj = model

                # If expected columns were found and X_user_enc has them, pass numpy array
                if expected_cols is not None:
                    pred = model_obj.predict(X_user_enc.values)
                else:
                    # try passing DataFrame directly
                    pred = model_obj.predict(X_user_enc)

                st.write("### Prediction result")
                # If prediction is array-like
                if hasattr(pred, '__len__'):
                    st.write(pred[0])
                else:
                    st.write(pred)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.write("""Troubleshooting tips:
- Ensure your model pipeline expects the same features and preprocessing you are providing.
- If you used one-hot encoding during training, the saved model may expect the same dummy columns; consider saving the training columns along with the model (e.g., save a dict {'model': model, 'feature_names': feature_names}).
- Try uploading your full pipeline (preprocessing + estimator) so the pipeline can handle raw DataFrame inputs.
""").
- Try uploading your full pipeline (preprocessing + estimator) so the pipeline can handle raw DataFrame inputs.")

    st.write("---")
    st.caption("App updated to load an existing joblib model (or upload one) and use it for prediction. Run with: `streamlit run streamlit_boston_app.py`")

if __name__ == '__main__':
    main()
