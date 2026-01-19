import uuid
import time
import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ML Imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

app = FastAPI(title="AutoML Master Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model store with a simple cleanup mechanism
MODELS: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    file_url: Optional[str] = None
    csv_text: Optional[str] = None
    target_column: Optional[str] = None
    target: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    algorithm: Optional[str] = "auto"
    problem_type: str = "auto"

# --- CORE LOGIC: Parallel Training Helper ---
def train_single_model(algo, problem_type, X, y, cv):
    """Function to train one specific model and return its metrics."""
    model_id = str(uuid.uuid4())
    t0 = time.time()
    try:
        model = _build_model(problem_type, algo)
        metrics = {}
        
        # Cross Validation (Parallel friendly)
        if cv is not None:
            scoring = "accuracy" if problem_type == "classification" else "r2"
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            metrics[f"cv_{scoring}_mean"] = float(cv_scores.mean())
            metrics[f"cv_{scoring}_std"] = float(cv_scores.std())
        
        # Final Fit
        model.fit(X, y)
        preds = model.predict(X)
        inf_t = (time.time() - t0) * 1000
        
        # Feature Importance
        fi = _feature_importance(model, X)
        
        return {
            "modelId": model_id,
            "algorithm": algo,
            "status": "ok",
            "metrics": metrics,
            "durationSec": time.time() - t0,
            "inferenceMs": inf_t,
            "featureImportance": fi,
            "model_obj": model # Temporarily hold to store in global dict
        }
    except Exception as e:
        return {"algorithm": algo, "status": "error", "message": str(e)}

# -------------------- REWRITTEN TRAIN ENDPOINT --------------------

@app.post("/train")
async def train(req: TrainRequest):
    start_total = time.time()
    
    # 1. Data Ingestion
    csv_data, err = _get_csv_text(req.file_url, req.csv_text)
    if err: return err
    
    target = req.target or req.target_column
    if not csv_data or not target:
        return {"error": "Missing input", "message": "Target and Data are required."}

    df = pd.read_csv(StringIO(csv_data))
    if target not in df.columns:
        return {"error": "Column not found", "target": target}

    # 2. Preprocessing
    X_raw = df[req.feature_columns] if req.feature_columns else df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X_raw, drop_first=True)
    
    problem_type = _safe_problem_type(y, req.problem_type)
    cv, _ = _safe_cv(problem_type, y, len(df))

    # 3. Parallel Execution
    candidates = ["dummy", "logistic", "decision_tree", "random_forest", "gradient_boosting"] if problem_type == "classification" else ["dummy", "linear", "decision_tree", "random_forest", "gradient_boosting"]
    
    if req.algorithm and req.algorithm != "auto":
        candidates = [req.algorithm]

    leaderboard = []
    # Use ThreadPoolExecutor to train all models at once
    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = [executor.submit(train_single_model, algo, problem_type, X, y, cv) for algo in candidates]
        for f in futures:
            res = f.result()
            if res["status"] == "ok":
                # Store model in global memory
                MODELS[res["modelId"]] = {
                    "model": res.pop("model_obj"),
                    "columns": X.columns.tolist(),
                    "problemType": problem_type
                }
            leaderboard.append(res)

    # 4. Final Best Model Selection
    ok_models = [m for m in leaderboard if m["status"] == "ok"]
    if not ok_models: return {"error": "Training failed for all models"}
    
    metric_key = "cv_accuracy_mean" if problem_type == "classification" else "cv_r2_mean"
    best_model = max(ok_models, key=lambda x: x["metrics"].get(metric_key, 0))

    return {
        "status": "success",
        "problemType": problem_type,
        "bestModel": best_model,
        "leaderboard": leaderboard,
        "totalTime": time.time() - start_total
    }

# (Keep your existing _build_model, _safe_cv, etc. helpers here)
