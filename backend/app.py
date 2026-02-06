import uuid
import time
import os
import pickle
import base64
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient

# Scientific ML Suite
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

app = FastAPI(title="Emergent AI Master Pro", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Connection
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://mongodb:27017/")
try:
    mongo_client = MongoClient(MONGO_URL)
    db = mongo_client["automl_db"]
    models_collection = db["models"]
except Exception:
    db = None

MODELS: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    csv_text: Optional[str] = None
    file_url: Optional[str] = None
    target_column: str
    problem_type: str = "auto"

class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]

# ==================== CORE ENGINEERING ====================

def build_universal_preprocessor(X):
    """Professional Feature Engineering for Numbers, Categories, and Text."""
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols, text_cols = [], []
    
    for col in X.select_dtypes(include=['object']).columns:
        if X[col].astype(str).str.len().mean() > 25: 
            text_cols.append(col)
        else: 
            cat_cols.append(col)

    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()) # Robust to outliers in datasets
        ]), num_cols))
        
    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols))
    
    for i, col in enumerate(text_cols):
        # NLP Track with bigrams for phrases like "Science Fiction"
        transformers.append((f'text_{i}', TfidfVectorizer(max_features=200, ngram_range=(1,2), stop_words='english'), col))

    return ColumnTransformer(transformers=transformers)

def train_single_model(algo_name, model_obj, X, y, problem_type):
    """Worker for Parallel Training Leaderboard."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        
        score = r2_score(y_test, y_pred) if problem_type == "regression" else accuracy_score(y_test, y_pred)
        return {"algo": algo_name, "model": model_obj, "score": score, "status": "ok"}
    except Exception as e:
        return {"algo": algo_name, "status": "error", "msg": str(e)}

# ==================== API ENDPOINTS ====================

@app.post("/api/train")
async def train(req: TrainRequest):
    # 1. Data Ingestion
    if req.csv_text: data = req.csv_text
    elif req.file_url: data = requests.get(req.file_url).text
    else: raise HTTPException(400, "No data provided")

    df = pd.read_csv(StringIO(data))
    
    # 2. Leakage Shield: Protects accuracy from being artificially inflated
    to_drop = [c for c in df.columns if any(k in c.lower() for k in ['id', 'uuid', 'date', 'timestamp']) and c != req.target_column]
    df = df.drop(columns=to_drop)

    X = df.drop(columns=[req.target_column])
    y = df[req.target_column]

    # 3. Dynamic Problem Detection
    is_regression = (y.dtype == 'float64' or y.nunique() > 10)
    problem = "regression" if is_regression else "classification"
    
    preprocessor = build_universal_preprocessor(X)
    
    # 4. Multi-Model Parallel Leaderboard
    algos = {
        "GradientBoosting": GradientBoostingRegressor() if is_regression else GradientBoostingClassifier(),
        "RandomForest": RandomForestRegressor() if is_regression else RandomForestClassifier(),
        "LinearModel": LinearRegression() if is_regression else LogisticRegression()
    }
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for name, model in algos.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            futures.append(executor.submit(train_single_model, name, pipe, X, y, problem))
        results = [f.result() for f in futures if f.result()["status"] == "ok"]

    # 5. Selection of Best Model
    best_res = max(results, key=lambda x: x["score"])
    model_id = str(uuid.uuid4())
    
    # 6. Persistence (Memory + MongoDB)
    model_data = {
        "model": best_res["model"],
        "problem": problem,
        "features": X.columns.tolist(),
        "score": best_res["score"],
        "algo": best_res["algo"]
    }
    MODELS[model_id] = model_data
    
    if db is not None:
        models_collection.insert_one({
            "modelId": model_id,
            "problem": problem,
            "score": best_res["score"],
            "model_bin": base64.b64encode(pickle.dumps(best_res["model"])).decode('utf-8')
        })

    return {
        "status": "success", 
        "modelId": model_id, 
        "accuracy": f"{round(best_res['score']*100, 2)}%",
        "algorithmUsed": best_res["algo"]
    }

@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Predict API with Scientific Confidence Scoring."""
    if req.model_id not in MODELS: raise HTTPException(404, "Model not found")
    
    model_info = MODELS[req.model_id]
    input_df = pd.DataFrame(req.data)
    
    # Ensure all features exist
    for col in model_info["features"]:
        if col not in input_df.columns: input_df[col] = np.nan

    preds = model_info["model"].predict(input_df[model_info["features"]])
    
    # Confidence Logic: Calculates reliability of the prediction
    confidence = "High"
    if model_info["problem"] == "classification" and hasattr(model_info["model"], "predict_proba"):
        probs = model_info["model"].predict_proba(input_df[model_info["features"]])
        avg_max_prob = np.mean([np.max(p) for p in probs])
        if avg_max_prob < 0.7: confidence = "Low"
        elif avg_max_prob < 0.85: confidence = "Medium"

    return {
        "predictions": preds.tolist(),
        "confidence": confidence,
        "modelType": model_info["problem"]
    }

@app.get("/api/models/{model_id}/download")
async def download(model_id: str):
    """Allows production export of models."""
    if model_id not in MODELS: raise HTTPException(404, "Model not found")
    bytes_io = BytesIO(pickle.dumps(MODELS[model_id]["model"]))
    return StreamingResponse(bytes_io, media_type="application/octet-stream", 
                             headers={"Content-Disposition": f"attachment; filename=model_{model_id[:8]}.pkl"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
