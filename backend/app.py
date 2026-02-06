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

# Professional ML Suite
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

app = FastAPI(title="Emergent AI Master Pro", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
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

# ==================== CORE ML LOGIC ====================

def build_universal_preprocessor(X):
    """Handles Numbers, Categories, and NLP Text for high accuracy."""
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
            ('scaler', RobustScaler()) # Better than StandardScaler for real-world data
        ]), num_cols))
        
    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols))
    
    for i, col in enumerate(text_cols):
        # NLP: Extracts 200 features including phrases (bigrams)
        transformers.append((f'text_{i}', TfidfVectorizer(max_features=200, ngram_range=(1,2)), col))

    return ColumnTransformer(transformers=transformers)

def run_training_worker(name, model, X, y, is_regression):
    """Trains individual models for the Leaderboard."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test)) if is_regression else accuracy_score(y_test, model.predict(X_test))
        return {"name": name, "model": model, "score": score, "status": "ok"}
    except Exception as e:
        return {"name": name, "status": "error", "msg": str(e)}

# ==================== API ENDPOINTS ====================

@app.post("/api/train")
async def train(req: TrainRequest):
    # Data Loading
    if req.csv_text: data = req.csv_text
    elif req.file_url: data = requests.get(req.file_url).text
    else: raise HTTPException(400, "No data provided")

    df = pd.read_csv(StringIO(data))
    
    # Leakage Prevention: Auto-removes IDs and Dates
    to_drop = [c for c in df.columns if any(k in c.lower() for k in ['id', 'uuid', 'date', 'added']) and c != req.target_column]
    df = df.drop(columns=to_drop)

    X = df.drop(columns=[req.target_column])
    y = df[req.target_column]

    is_reg = (y.dtype == 'float64' or y.nunique() > 10)
    problem = "regression" if is_reg else "classification"
    preprocessor = build_universal_preprocessor(X)
    
    # Parallel Leaderboard: Tests 3 Algos to ensure accuracy
    algos = {
        "GradientBoost": GradientBoostingRegressor() if is_reg else GradientBoostingClassifier(),
        "RandomForest": RandomForestRegressor() if is_reg else RandomForestClassifier(),
        "Linear": LinearRegression() if is_reg else LogisticRegression(max_iter=1000)
    }
    
    futures = []
    with ThreadPoolExecutor() as executor:
        for n, m in algos.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', m)])
            futures.append(executor.submit(run_training_worker, n, pipe, X, y, is_reg))
    
    results = [f.result() for f in futures if f.result()["status"] == "ok"]
    if not results: raise HTTPException(500, "Training failed")

    best = max(results, key=lambda x: x["score"])
    model_id = str(uuid.uuid4())
    
    # Save Model
    MODELS[model_id] = {"model": best["model"], "problem": problem, "features": X.columns.tolist()}
    
    if db is not None:
        models_collection.insert_one({
            "modelId": model_id,
            "accuracy": best["score"],
            "bin": base64.b64encode(pickle.dumps(best["model"])).decode('utf-8')
        })

    return {"status": "success", "modelId": model_id, "accuracy": f"{round(best['score']*100, 2)}%", "type": problem}

@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Predict API with Confidence scoring."""
    if req.model_id not in MODELS: raise HTTPException(404, "Model not found")
    
    m_info = MODELS[req.model_id]
    input_df = pd.DataFrame(req.data)
    
    # Alignment check
    for col in m_info["features"]:
        if col not in input_df.columns: input_df[col] = np.nan

    preds = m_info["model"].predict(input_df[m_info["features"]])
    
    return {
        "predictions": preds.tolist(),
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
