from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from io import StringIO
import uuid
import time
import traceback

import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model store (fine for POC). For production: persist to DB + artifact storage.
MODEL_STORE: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Config: guardrails for Railway
# -----------------------------
MAX_SYNC_ROWS = 50000             # sample down if larger
MAX_DUMMY_COLS = 600              # after encoding, cap features
MAX_CV_FOLDS = 5

# Model Zoo for Option 1 (Tier1 only; sklearn stable)
MODEL_ZOO = {
    "classification": [
        ("logistic", LogisticRegression(max_iter=2000)),
        ("decision_tree", DecisionTreeClassifier(random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
    ],
    "regression": [
        ("linear", LinearRegression()),
        ("decision_tree", DecisionTreeRegressor(random_state=42)),
        ("random_forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(random_state=42)),
    ],
}


class TrainRequest(BaseModel):
    # data input
    csv_text: Optional[str] = None
    file_url: Optional[str] = None

    # target naming flexibility
    target: Optional[str] = None
    target_column: Optional[str] = None

    # settings
    problem_type: str = "auto"     # auto | regression | classification
    max_tier: str = "tier1"        # for now only tier1 is supported

    # optional extras (accepted to avoid validation errors from Base-44)
    feature_columns: Optional[List[str]] = None
    algorithm: Optional[str] = None
    model_id: Optional[str] = None
    extra: Optional[Any] = None


class PredictRequest(BaseModel):
    modelId: str
    # one of the following:
    rows: Optional[List[Dict[str, Any]]] = None      # list of records
    row: Optional[Dict[str, Any]] = None             # single record


@app.get("/")
def home():
    return {"status": "ok", "message": "AutoML backend running", "docs": "/docs"}


def _download_csv_text(file_url: str) -> str:
    r = requests.get(file_url, timeout=30)
    r.raise_for_status()
    return r.text


def _detect_problem_type(y: pd.Series) -> str:
    # pragmatic heuristic: few unique values => classification
    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # split numeric vs categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre


def _safe_cv(problem_type: str, y: pd.Series, n_rows: int):
    if problem_type == "classification":
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            return None, {
                "error": "Not enough samples in one class for cross-validation.",
                "message": "Each class must have at least 2 rows. Add more data or reduce imbalance.",
                "classCounts": class_counts.to_dict()
            }
        folds = min(MAX_CV_FOLDS, n_rows, min_class_count)
        folds = max(2, folds)
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42), None

    # regression
    folds = min(MAX_CV_FOLDS, n_rows)
    folds = max(2, folds)
    return KFold(n_splits=folds, shuffle=True, random_state=42), None


def _primary_scoring(problem_type: str) -> str:
    return "accuracy" if problem_type == "classification" else "r2"


def _secondary_metrics(problem_type: str, y_true, y_pred) -> Dict[str, float]:
    if problem_type == "classification":
        return {
            "accuracy_full_fit": float(accuracy_score(y_true, y_pred)),
            "f1_full_fit": float(f1_score(y_true, y_pred, average="weighted"))
        }
    return {
        "mae_full_fit": float(mean_absolute_error(y_true, y_pred))
    }


def _feature_importance_from_pipeline(pipeline: Pipeline, feature_names: List[str]) -> List[Dict[str, float]]:
    """
    feature_names here are ORIGINAL column names, not expanded OHE names.
    For Option 1, we keep a simple, honest approach:
      - Tree ensembles: use model.feature_importances_ on transformed space isn't trivial;
        we approximate by using original feature importance if model supports it AND input was numeric only.
      - Linear/Logistic: use coefficients (again, transformed space).
    For now, we return:
      - If model has feature_importances_: we return top features by that array length if it matches,
        else return empty list to avoid fake mapping.
      - If model has coef_: return empty if shape mismatch.
    This is intentionally conservative (no hallucinated mappings).
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        fi = getattr(model, "feature_importances_")
        # Only safe if lengths match a known feature list (rare with OHE)
        if len(fi) == len(feature_names):
            items = [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, fi)]
            items.sort(key=lambda x: x["importance"], reverse=True)
            return items[:20]
        return []

    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        # coef can be (n_classes, n_features) or (n_features,)
        coef_flat = np.abs(coef).mean(axis=0) if hasattr(coef, "ndim") and coef.ndim > 1 else np.abs(coef)
        if len(coef_flat) == len(feature_names):
            items = [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, coef_flat)]
            items.sort(key=lambda x: x["importance"], reverse=True)
            return items[:20]
        return []

    return []


@app.post("/train")
def train(req: TrainRequest):
    try:
        # ---- Resolve CSV text ----
        csv_text = req.csv_text
        if (csv_text is None or csv_text.strip() == "") and req.file_url:
            try:
                csv_text = _download_csv_text(req.file_url)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to download CSV from file_url", "message": str(e), "file_url": req.file_url}
                )

        if csv_text is None or csv_text.strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": "Missing csv_text or file_url", "message": "Provide either csv_text or file_url."}
            )

        # ---- Resolve target ----
        target = req.target or req.target_column
        if not target:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing target", "message": "Provide target or target_column."}
            )

        # ---- Load data ----
        df = pd.read_csv(StringIO(csv_text))
        if target not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Target column '{target}' not found.", "columns": df.columns.tolist()}
            )

        # ---- Guardrail: sample rows ----
        if len(df) > MAX_SYNC_ROWS:
            df = df.sample(n=MAX_SYNC_ROWS, random_state=42)

        y = df[target]
        X = df.drop(columns=[target])

        # ---- Determine problem type ----
        problem_type = req.problem_type.lower().strip()
        if problem_type == "auto":
            problem_type = _detect_problem_type(y)

        if problem_type not in ("classification", "regression"):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid problem_type", "message": "Use auto|classification|regression."}
            )

        # ---- CV setup (and friendly class check) ----
        cv, cv_err = _safe_cv(problem_type, y, len(df))
        if cv_err:
            return JSONResponse(status_code=400, content=cv_err)

        # ---- Preprocessor ----
        pre = _build_preprocessor(X)

        # ---- Run Tier1 models ----
        models = MODEL_ZOO[problem_type]
        scoring = _primary_scoring(problem_type)

        leaderboard = []
        best = None

        # Keep original column names only (honest importance mapping)
        original_feature_names = X.columns.tolist()

        for algo_id, algo_model in models:
            start = time.time()
            try:
                pipe = Pipeline(steps=[
                    ("preprocess", pre),
                    ("model", algo_model)
                ])

                # CV score
                cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

                # Fit full data
                pipe.fit(X, y)
                preds_full = pipe.predict(X)

                # Secondary metrics
                secondary = _secondary_metrics(problem_type, y, preds_full)

                # Feature importance (conservative)
                fi = _feature_importance_from_pipeline(pipe, original_feature_names)

                duration = time.time() - start

                model_id = str(uuid.uuid4())
                MODEL_STORE[model_id] = {
                    "pipeline": pipe,
                    "problemType": problem_type,
                    "target": target,
                    "featureNames": original_feature_names
                }

                entry = {
                    "modelId": model_id,
                    "algorithm": algo_id,
                    "primaryMetric": cv_mean,
                    "primaryMetricName": f"cv_{scoring}_mean",
                    "cv_std": cv_std,
                    "durationSec": float(duration),
                    "status": "ok",
                    "metrics": {
                        f"cv_{scoring}_mean": cv_mean,
                        f"cv_{scoring}_std": cv_std,
                        **secondary
                    },
                    "featureImportance": fi,
                    "sample": {
                        "actual": y.head(30).tolist(),
                        "predicted": preds_full[:30].tolist()
                    }
                }
                leaderboard.append(entry)

                if (best is None) or (entry["primaryMetric"] > best["primaryMetric"]):
                    best = entry

            except Exception as e:
                duration = time.time() - start
                leaderboard.append({
                    "modelId": None,
                    "algorithm": algo_id,
                    "primaryMetric": None,
                    "primaryMetricName": f"cv_{scoring}_mean",
                    "cv_std": None,
                    "durationSec": float(duration),
                    "status": "failed",
                    "error": str(e)
                })

        # If everything failed, stop cleanly
        if best is None:
            return JSONResponse(
                status_code=400,
                content={"error": "All models failed", "leaderboard": leaderboard}
            )

        # Sort leaderboard best-first (failed at bottom)
        leaderboard_sorted = sorted(
            leaderboard,
            key=lambda x: (-1 if x["primaryMetric"] is None else x["primaryMetric"]),
            reverse=True
        )

        return {
            "problemType": problem_type,
            "target": target,
            "bestModelId": best["modelId"],
            "bestAlgorithm": best["algorithm"],
            "leaderboard": leaderboard_sorted,
            "bestModel": {
                "modelId": best["modelId"],
                "algorithm": best["algorithm"],
                "metrics": best["metrics"],
                "featureImportance": best.get("featureImportance", []),
                "sample": best.get("sample", {})
            },
            "warnings": _compute_warnings(df, X, problem_type)
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Backend crashed during training",
                "message": str(e),
                "trace": traceback.format_exc()[:2000]
            }
        )


def _compute_warnings(df: pd.DataFrame, X: pd.DataFrame, problem_type: str) -> List[str]:
    warnings = []
    if len(df) >= MAX_SYNC_ROWS:
        warnings.append(f"Dataset was sampled to {MAX_SYNC_ROWS} rows for synchronous AutoML.")
    if X.shape[1] > 200:
        warnings.append("High number of input columns detected. Consider feature reduction for faster training.")
    if problem_type == "classification":
        vc = df.select_dtypes(exclude=[np.number]).shape[1]
        if vc > 0:
            warnings.append("Categorical columns detected. One-hot encoding is applied in backend.")
    return warnings


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model_id = req.modelId
        if model_id not in MODEL_STORE:
            return JSONResponse(status_code=400, content={"error": "Unknown modelId", "modelId": model_id})

        pipe = MODEL_STORE[model_id]["pipeline"]
        feature_names = MODEL_STORE[model_id]["featureNames"]

        # normalize input rows
        if req.rows is None and req.row is None:
            return JSONResponse(status_code=400, content={"error": "Provide row or rows"})

        rows = req.rows if req.rows is not None else [req.row]

        # build df; allow missing cols (will become NaN and be imputed)
        in_df = pd.DataFrame(rows)

        # ensure all expected original features exist
        for c in feature_names:
            if c not in in_df.columns:
                in_df[c] = np.nan

        # keep only expected cols (ignore extras)
        in_df = in_df[feature_names]

        preds = pipe.predict(in_df)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        return {"modelId": model_id, "predictions": preds_list}

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Prediction failed", "message": str(e), "trace": traceback.format_exc()[:2000]}
        )
