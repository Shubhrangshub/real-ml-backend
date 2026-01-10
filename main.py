import time
import uuid
import requests
import pandas as pd
import numpy as np
from io import StringIO
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# ML Components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

# --- App Initialization ---
app = FastAPI(title="Gemini ML Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Registry for Trained Models
MODELS: Dict[str, Dict[str, Any]] = {}

# --- Schemas ---
class TrainRequest(BaseModel):
    file_url: Optional[str] = None
    csv_text: Optional[str] = None
    target: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    algorithm: str = "auto"
    problem_type: str = "auto"

class PredictRequest(BaseModel):
    modelId: str
    file_url: Optional[str] = None
    csv_text: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    feature_columns: Optional[List[str]] = None

# --- Internal Helpers ---
def _get_csv_text(file_url, csv_text):
    if csv_text and csv_text.strip(): return csv_text, None
    if file_url:
        try:
            r = requests.get(file_url, timeout=15)
            r.raise_for_status()
            return r.content.decode('utf-8-sig', errors='replace'), None
        except Exception as e:
            return None, {"error": "Download failed", "message": str(e)}
    return None, None

def _align_to_training_columns(X_raw: pd.DataFrame, train_cols: List[str]) -> pd.DataFrame:
    X = pd.get_dummies(X_raw, drop_first=True)
    return X.reindex(columns=train_cols, fill_value=0)

def _build_model(problem_type: str, algo: str):
    def make_pipe(est):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", est)
        ])
    
    if problem_type == "classification":
        mapping = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic": LogisticRegression(max_iter=2000),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42)
        }
    else:
        mapping = {
            "dummy": DummyRegressor(strategy="mean"),
            "linear": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42)
        }
    return make_pipe(mapping[algo])

def _safe_problem_type(y, requested):
    if requested and requested != "auto": return requested
    return "classification" if y.nunique() < 20 else "regression"

def _safe_cv(problem_type, y, n_rows):
    if n_rows < 5: return None, "Too few rows for CV."
    if problem_type == "classification":
        vc = y.value_counts()
        if vc.min() < 2: return KFold(n_splits=3, shuffle=True), "Small class sizes: using KFold."
        return StratifiedKFold(n_splits=min(5, vc.min()), shuffle=True), None
    return KFold(n_splits=min(5, n_rows), shuffle=True), None

def _feature_importance(model, cols):
    try:
        core = model.named_steps["model"]
        if hasattr(core, "feature_importances_"): imp = core.feature_importances_
        elif hasattr(core, "coef_"): imp = np.abs(core.coef_).ravel()
        else: return []
        return sorted([{"feature": f, "importance": float(v)} for f, v in zip(cols, imp)], 
                      key=lambda x: x["importance"], reverse=True)[:15]
    except: return []

def _score_primary(p_type, m):
    if p_type == "classification":
        return "cv_accuracy", m.get("cv_acc", 0.0), m.get("cv_std", 0.0)
    return "cv_r2", m.get("cv_r2", -1e9), m.get("cv_std", 0.0)

# --- API Endpoints ---
@app.get("/")
def root(): return {"ok": True, "status": "ML backend ready"}

@app.post("/train")
def train(req: TrainRequest):
    start_all = time.time()
    warnings = []
    
    # 1. Load Data
    csv_text, err = _get_csv_text(req.file_url, req.csv_text)
    if err: return err
    target = req.target or req.target_column
    if not csv_text or not target: return {"error": "Missing data or target"}
    
    df = pd.read_csv(StringIO(csv_text), low_memory=False).dropna(how='all')
    if target not in df.columns: return {"error": f"Target '{target}' not in data"}

    # 2. Prepare X, y
    X_raw = df[req.feature_columns].copy() if req.feature_columns else df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Clean target NaNs
    if y.isnull().any():
        mask = y.notnull()
        X_raw, y = X_raw[mask], y[mask]
        warnings.append("Rows with empty targets were dropped.")

    problem_type = _safe_problem_type(y, req.problem_type)
    X = pd.get_dummies(X_raw, drop_first=True)
    cv, cv_warn = _safe_cv(problem_type, y, len(X))
    if cv_warn: warnings.append(cv_warn)

    candidates = [req.algorithm] if req.algorithm != "auto" else (
        ["dummy", "logistic", "decision_tree", "random_forest", "gradient_boosting"] 
        if problem_type == "classification" else ["dummy", "linear", "decision_tree", "random_forest", "gradient_boosting"]
    )

    leaderboard = []
    for algo in candidates:
        m_id, t0 = str(uuid.uuid4()), time.time()
        metrics, status, inf_ms = {}, "ok", 0.0
        try:
            model = _build_model(problem_type, algo)
            if cv:
                score_key = "accuracy" if problem_type == "classification" else "r2"
                cv_res = cross_val_score(model, X, y, cv=cv, scoring=score_key)
                metrics[f"cv_{score_key}"] = float(cv_res.mean())
                metrics["cv_std"] = float(cv_res.std())

            model.fit(X, y)
            t_inf = time.time()
            preds = model.predict(X)
            inf_ms = (time.time() - t_inf) * 1000
            
            MODELS[m_id] = {"model": model, "columns": X.columns.tolist(), "problemType": problem_type}
        except Exception as e:
            status, metrics = "error", {"error": str(e)}

        p_name, p_val, p_std = _score_primary(problem_type, metrics)
        leaderboard.append({
            "modelId": m_id, "algorithm": algo, "primaryMetric": p_val, "status": status,
            "metrics": metrics, "inferenceMs": inf_ms, "durationSec": time.time() - t0,
            "featureImportance": _feature_importance(model, X.columns.tolist()) if status == "ok" else []
        })

    ok_rows = [r for r in leaderboard if r["status"] == "ok"]
    if not ok_rows: return {"error": "All models failed", "leaderboard": leaderboard}
    best = sorted(ok_rows, key=lambda r: r["primaryMetric"], reverse=True)[0]

    return {
        "bestModelId": best["modelId"], "problemType": problem_type, 
        "leaderboard": leaderboard, "warnings": warnings, "durationTotal": time.time() - start_all
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if req.modelId not in MODELS: return {"error": "Model not found"}
    entry = MODELS[req.modelId]
    
    csv_text, err = _get_csv_text(req.file_url, req.csv_text)
    if err and not req.rows: return err
    
    df_new = pd.DataFrame(req.rows) if req.rows else pd.read_csv(StringIO(csv_text))
    X_new_raw = df_new[req.feature_columns].copy() if req.feature_columns else df_new.copy()
    
    try:
        X_new = _align_to_training_columns(X_new_raw, entry["columns"])
        t0 = time.time()
        preds = entry["model"].predict(X_new)
        res = {
            "modelId": req.modelId, "predictions": preds.tolist(), 
            "timeMs": (time.time() - t0) * 1000
        }
        if entry["problemType"] == "classification" and hasattr(entry["model"], "predict_proba"):
            res["probabilities"] = entry["model"].predict_proba(X_new).tolist()
        return res
    except Exception as e: return {"error": "Prediction failed", "message": str(e)}
