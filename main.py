from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from io import StringIO
import uuid
import time
import requests

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score


app = FastAPI(title="Real ML Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory model store (simple demo)
MODELS: Dict[str, Dict[str, Any]] = {}


# ----------- Request schema (Base-44 + Swagger compatible) -----------

class TrainRequest(BaseModel):
    # Base-44 style
    file_url: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    algorithm: Optional[str] = "auto"   # optional single-algo request

    # Swagger/manual style
    csv_text: Optional[str] = None
    target: Optional[str] = None

    # common
    problem_type: str = "auto"  # "auto" | "classification" | "regression"

class PredictRequest(BaseModel):
    modelId: str

    # Base-44 style
    file_url: Optional[str] = None
    feature_columns: Optional[List[str]] = None

    # Swagger/manual style
    csv_text: Optional[str] = None

    # Optional: if user sends raw JSON rows instead of CSV
    rows: Optional[List[Dict[str, Any]]] = None


# ----------- Helpers -----------

def _safe_problem_type(y: pd.Series, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    # heuristic: small unique -> classification
    # (you can make this smarter later)
    return "classification" if y.nunique() < 20 else "regression"


def _safe_cv(problem_type: str, y: pd.Series, n_rows: int):
    """
    Return a CV splitter that won't crash on tiny datasets.
    """
    if n_rows < 4:
        return None, "Dataset too small for CV. Need at least 4 rows."

    if problem_type == "classification":
        counts = y.value_counts()
        min_class = int(counts.min()) if len(counts) else 0
        if min_class < 2:
            return None, (
                "Not enough samples in one class for cross-validation. "
                "Each class must have at least 2 rows."
            )
        folds = min(5, n_rows, min_class)
        if folds < 2:
            folds = 2
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42), None

    # regression
    folds = min(5, n_rows)
    if folds < 2:
        folds = 2
    return KFold(n_splits=folds, shuffle=True, random_state=42), None


def _feature_importance(model, X: pd.DataFrame) -> List[Dict[str, float]]:
    """
    Return top feature importances/coeffs when available.
    """
    try:
        # Pipeline support
        if hasattr(model, "named_steps"):
            core = model.named_steps.get("model", None)
        else:
            core = model

        if core is None:
            return []

        if hasattr(core, "feature_importances_"):
            imp = core.feature_importances_
        elif hasattr(core, "coef_"):
            coef = core.coef_
            imp = np.abs(coef).ravel()
        else:
            return []

        pairs = [{"feature": f, "importance": float(v)} for f, v in zip(X.columns, imp)]
        pairs.sort(key=lambda d: d["importance"], reverse=True)
        return pairs[:20]
    except Exception:
        return []


def _build_model(problem_type: str, algo: str):
    """
    Return a sklearn estimator or Pipeline for given algo.
    """
    if problem_type == "classification":
        if algo == "dummy":
            return DummyClassifier(strategy="most_frequent")
        if algo == "logistic":
            # scaler helps on numeric; use with_mean=False to avoid sparse issues
            return Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=5000))
            ])
        if algo == "decision_tree":
            return DecisionTreeClassifier(random_state=42)
        if algo == "random_forest":
            return RandomForestClassifier(n_estimators=300, random_state=42)
        if algo == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        raise ValueError(f"Unknown classification algo: {algo}")

    # regression
    if algo == "dummy":
        return DummyRegressor(strategy="mean")
    if algo == "linear":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LinearRegression())
        ])
    if algo == "decision_tree":
        return DecisionTreeRegressor(random_state=42)
    if algo == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    if algo == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42)
    raise ValueError(f"Unknown regression algo: {algo}")


def _score_primary(problem_type: str, metrics: Dict[str, Any]) -> (str, float, float):
    """
    Return (primaryMetricName, primaryMetricValue, cv_std)
    """
    if problem_type == "classification":
        return "cv_accuracy_mean", float(metrics.get("cv_accuracy_mean", 0.0)), float(metrics.get("cv_accuracy_std", 0.0))
    return "cv_r2_mean", float(metrics.get("cv_r2_mean", -1e9)), float(metrics.get("cv_r2_std", 0.0))

def _align_to_training_columns(X_new: pd.DataFrame, train_cols: List[str]) -> pd.DataFrame:
    # one-hot encode
    X_new = pd.get_dummies(X_new, drop_first=True)

    # add missing cols as 0
    for c in train_cols:
        if c not in X_new.columns:
            X_new[c] = 0

    # drop extra cols
    X_new = X_new[train_cols]
    return X_new


# ----------- API -----------

@app.get("/")
def root():
    return {"ok": True, "message": "ML backend running", "docs": "/docs"}


@app.post("/train")
def train(req: TrainRequest):
    start_all = time.time()
    warnings: List[str] = []

    # --- normalize input: get csv_text ---
    csv_text = req.csv_text
    if not csv_text and req.file_url:
        try:
            r = requests.get(req.file_url, timeout=30)
            r.raise_for_status()
            csv_text = r.text
        except Exception as e:
            return {
                "error": "Failed to download CSV from file_url",
                "message": str(e),
                "file_url": req.file_url
            }

    @app.post("/predict")
def predict(req: PredictRequest):
    # 1) Validate model
    if req.modelId not in MODELS:
        return {"error": "Model not found", "message": "Invalid modelId", "modelId": req.modelId}

    entry = MODELS[req.modelId]
    model = entry["model"]
    train_cols = entry["columns"]
    problem_type = entry.get("problemType", "auto")

    # 2) Get input data (csv_text OR file_url OR rows)
    csv_text = req.csv_text
    if not csv_text and req.file_url:
        try:
            r = requests.get(req.file_url, timeout=30)
            r.raise_for_status()
            csv_text = r.text
        except Exception as e:
            return {"error": "Failed to download CSV from file_url", "message": str(e), "file_url": req.file_url}

    # 3) Build X_new
    try:
        if req.rows:
            df_new = pd.DataFrame(req.rows)
        else:
            if not csv_text:
                return {"error": "Missing data", "message": "Provide csv_text or file_url or rows"}
            df_new = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        return {"error": "Invalid input data", "message": str(e)}

    # Optional: respect feature_columns if Base-44 sends selected features
    if req.feature_columns:
        missing = [c for c in req.feature_columns if c not in df_new.columns]
        if missing:
            return {"error": "Feature columns missing", "missing": missing, "columns": df_new.columns.tolist()}
        X_new_raw = df_new[req.feature_columns].copy()
    else:
        X_new_raw = df_new.copy()

    # 4) Align with training columns
    try:
        X_new = _align_to_training_columns(X_new_raw, train_cols)
    except Exception as e:
        return {"error": "Failed to align features", "message": str(e)}

    # 5) Predict
    try:
        preds = model.predict(X_new)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        out = {
            "modelId": req.modelId,
            "problemType": problem_type,
            "predictions": preds_list
        }

        # Probabilities for classification (if supported)
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)
            out["proba"] = proba.tolist()

        return out

    except Exception as e:
        return {"error": "Prediction failed", "message": str(e)}

    # --- normalize target ---
    target = req.target or req.target_column
    if not csv_text:
        return {"error": "Missing data", "message": "Provide csv_text or file_url"}
    if not target:
        return {"error": "Missing target", "message": "Provide target or target_column"}

    # --- load df ---
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        return {"error": "Invalid CSV", "message": str(e)}

    if target not in df.columns:
        return {"error": "Target not found", "message": f"'{target}' not in CSV columns", "columns": df.columns.tolist()}

    # --- build X,y ---
    if req.feature_columns:
        missing = [c for c in req.feature_columns if c not in df.columns]
        if missing:
            return {"error": "Feature columns missing", "message": "Some selected features not in CSV", "missing": missing}
        X = df[req.feature_columns].copy()
    else:
        X = df.drop(columns=[target]).copy()

    y = df[target].copy()

    # basic cleaning
    X = pd.get_dummies(X, drop_first=True)
    n_rows = len(df)

    problem_type = _safe_problem_type(y, req.problem_type)

    # --- CV splitter ---
    cv, cv_warn = _safe_cv(problem_type, y, n_rows)
    if cv_warn:
        warnings.append(cv_warn)

    # --- algorithm set (Tier0/Tier1) ---
    if problem_type == "classification":
        candidates = ["dummy", "logistic", "decision_tree", "random_forest", "gradient_boosting"]
    else:
        candidates = ["dummy", "linear", "decision_tree", "random_forest", "gradient_boosting"]

    # if user requested one algorithm only
    if req.algorithm and req.algorithm != "auto":
        if req.algorithm not in candidates:
            return {
                "error": "Unsupported algorithm for this problem_type",
                "message": f"'{req.algorithm}' not in {candidates}",
                "problemType": problem_type
            }
        candidates = [req.algorithm]

    leaderboard = []

    for algo in candidates:
        model_id = str(uuid.uuid4())
        t0 = time.time()
        status = "ok"
        metrics: Dict[str, Any] = {}
        fi: List[Dict[str, float]] = []
        sample_actual = []
        sample_pred = []

        try:
            model = _build_model(problem_type, algo)

            # CV metrics
            if cv is not None:
                if problem_type == "classification":
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                    metrics["cv_accuracy_mean"] = float(cv_scores.mean())
                    metrics["cv_accuracy_std"] = float(cv_scores.std())
                else:
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                    metrics["cv_r2_mean"] = float(cv_scores.mean())
                    metrics["cv_r2_std"] = float(cv_scores.std())
            else:
                # no CV possible
                if problem_type == "classification":
                    metrics["cv_accuracy_mean"] = 0.0
                    metrics["cv_accuracy_std"] = 0.0
                else:
                    metrics["cv_r2_mean"] = -1e9
                    metrics["cv_r2_std"] = 0.0

            # full fit + sample preds
            model.fit(X, y)
            preds_full = model.predict(X)

            if problem_type == "classification":
                metrics["accuracy_full_fit"] = float(accuracy_score(y, preds_full))
                metrics["f1_full_fit"] = float(f1_score(y, preds_full, average="weighted"))
            else:
                metrics["r2_full_fit"] = float(r2_score(y, preds_full))
                metrics["mae_full_fit"] = float(mean_absolute_error(y, preds_full))

            fi = _feature_importance(model, X)

            sample_actual = y.head(30).tolist()
            sample_pred = preds_full[:30].tolist()

            # store trained model
            MODELS[model_id] = {"model": model, "columns": X.columns.tolist(), "problemType": problem_type}

        except Exception as e:
            status = "error"
            metrics = {"error": str(e)}

        duration = time.time() - t0

        primary_name, primary_val, cv_std = _score_primary(problem_type, metrics)

        leaderboard.append({
            "modelId": model_id,
            "algorithm": algo,
            "primaryMetric": primary_val,
            "primaryMetricName": primary_name,
            "cv_std": cv_std,
            "durationSec": float(duration),
            "status": status,
            "metrics": metrics,
            "featureImportance": fi,
            "sample": {"actual": sample_actual, "predicted": sample_pred}
        })

    # choose best (max primaryMetric)
    ok_rows = [r for r in leaderboard if r["status"] == "ok"]
    if not ok_rows:
        return {
            "error": "All models failed",
            "problemType": problem_type,
            "target": target,
            "leaderboard": leaderboard,
            "warnings": warnings
        }

    best = sorted(ok_rows, key=lambda r: r["primaryMetric"], reverse=True)[0]

    result = {
        "problemType": problem_type,
        "target": target,
        "bestModelId": best["modelId"],
        "bestAlgorithm": best["algorithm"],
        "leaderboard": leaderboard,
        "bestModel": {
            "modelId": best["modelId"],
            "algorithm": best["algorithm"],
            "metrics": best["metrics"],
            "featureImportance": best["featureImportance"],
            "sample": best["sample"]
        },
        "warnings": warnings,
        "durationSecTotal": float(time.time() - start_all)
    }
    return result
