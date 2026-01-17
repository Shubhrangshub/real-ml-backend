import time
import uuid
import requests
import pandas as pd
import numpy as np
import os
from io import StringIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
from sklearn.model_selection import cross_val_score

# --- App Initialization ---
app = FastAPI(title="Gemini ML Engine Pro", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (Note: Resets on Railway deploy/restart)
MODELS: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    file_url: Optional[str] = None
    csv_text: Optional[str] = None
    target_column: str  # Required
    algorithm: str = "auto"
    problem_type: str = "auto" # "classification", "regression", or "auto"

class PredictRequest(BaseModel):
    modelId: str
    file_url: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None

# --- Internal Core ---

def _fetch_data(file_url, csv_text):
    if csv_text: return csv_text
    if file_url:
        r = requests.get(file_url, timeout=20)
        r.raise_for_status()
        return r.text
    return None

def _get_pipeline(problem_type: str, algo: str):
    """Tiered Logic: Ensures lightweight but powerful models for Railway."""
    if problem_type == "classification":
        mapping = {
            "logistic": LogisticRegression(max_iter=1000, solver='liblinear'),
            "random_forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=50),
            "dummy": DummyClassifier(strategy="most_frequent")
        }
    else: # Regression
        mapping = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=50),
            "dummy": DummyRegressor(strategy="mean")
        }
    
    model = mapping.get(algo, mapping["random_forest"]) # Default to RF if unknown
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model)
    ])

# --- API Endpoints ---

@app.get("/")
def health():
    return {"status": "ready", "engine": "Gemini-ML-Pro", "port": os.getenv("PORT", "8000")}

@app.post("/train")
async def train(req: TrainRequest):
    try:
        data_str = _fetch_data(req.file_url, req.csv_text)
        if not data_str: return {"error": "No data source found"}
        
        df = pd.read_csv(StringIO(data_str)).dropna(subset=[req.target_column])
        
        X_raw = df.drop(columns=[req.target_column])
        y = df[req.target_column]

        # 1. Seamless Problem Detection
        p_type = req.problem_type
        if p_type == "auto":
            p_type = "classification" if y.nunique() < 15 else "regression"

        # 2. Automated Feature Engineering (Handling Text/Categories)
        X = pd.get_dummies(X_raw, drop_first=True)
        feature_cols = X.columns.tolist()

        # 3. Algorithm Selection (Auto-pick Tiers)
        if req.algorithm == "auto":
            algos = ["logistic", "random_forest"] if p_type == "classification" else ["linear", "random_forest"]
        else:
            algos = [req.algorithm]

        leaderboard = []
        for algo in algos:
            m_id = str(uuid.uuid4())
            pipe = _get_pipeline(p_type, algo)
            
            # Use R2 for Regression, Accuracy for Classification
            metric_name = "accuracy" if p_type == "classification" else "r2"
            scores = cross_val_score(pipe, X, y, cv=3, scoring=metric_name)
            
            pipe.fit(X, y)
            
            MODELS[m_id] = {"pipeline": pipe, "cols": feature_cols, "type": p_type}
            leaderboard.append({"modelId": m_id, "algo": algo, "score": round(float(scores.mean()), 4)})

        # Sort to give the best model back
        leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
        
        return {
            "status": "success",
            "bestModelId": leaderboard[0]["modelId"],
            "problemType": p_type,
            "leaderboard": leaderboard
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(req: PredictRequest):
    if req.modelId not in MODELS: raise HTTPException(status_code=404, detail="Model not found")
    
    m = MODELS[req.modelId]
    df_new = pd.DataFrame(req.rows) if req.rows else pd.read_csv(StringIO(_fetch_data(req.file_url, None)))
    
    # 4. Seamless Column Alignment (Fixes missing columns on the fly)
    X_new = pd.get_dummies(df_new).reindex(columns=m["cols"], fill_value=0)
    
    preds = m["pipeline"].predict(X_new)
    return {"modelId": req.modelId, "predictions": preds.tolist()}
