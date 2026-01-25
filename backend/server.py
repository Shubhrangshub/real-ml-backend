"""
AutoML Master Backend - Professional Machine Learning Training Platform

This FastAPI application provides enterprise-grade automated machine learning
capabilities with advanced text processing, data leakage prevention, and
scientifically rigorous validation.

Key Features:
    - Automatic data leakage detection and removal
    - Text processing with TF-IDF vectorization (bigrams)
    - Parallel model training with multiple algorithms
    - Robust cross-validation (ShuffleSplit for regression)
    - Model persistence with MongoDB
    - Real-time prediction API
    - Feature importance extraction

Author: AutoML Master Team
Version: 2.0 (Scientific Edition)
"""

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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient

# ML Imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, mean_squared_error

app = FastAPI(title="AutoML Master Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Setup
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
try:
    mongo_client = MongoClient(MONGO_URL)
    db = mongo_client["automl_db"]
    models_collection = db["models"]
    training_history_collection = db["training_history"]
except Exception as e:
    print(f"MongoDB connection warning: {e}")
    db = None

# Global model store (in-memory backup)
MODELS: Dict[str, Dict[str, Any]] = {}

# ==================== REQUEST/RESPONSE MODELS ====================

class TrainRequest(BaseModel):
    file_url: Optional[str] = None
    csv_text: Optional[str] = None
    target_column: Optional[str] = None
    target: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    algorithm: Optional[str] = "auto"
    problem_type: str = "auto"

class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]

# ==================== HELPER FUNCTIONS ====================

def _get_csv_text(file_url: Optional[str], csv_text: Optional[str]) -> tuple:
    """
    Fetch CSV data from URL or use provided text.
    
    Args:
        file_url: Optional URL to fetch CSV data from
        csv_text: Optional raw CSV text data
        
    Returns:
        tuple: (csv_data: str, error: dict or None)
        
    Example:
        >>> csv_data, err = _get_csv_text(None, "col1,col2\\n1,2")
        >>> if not err:
        >>>     df = pd.read_csv(StringIO(csv_data))
    """
    try:
        if csv_text:
            return csv_text, None
        elif file_url:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            return response.text, None
        else:
            return None, {"error": "No data provided", "message": "Provide either file_url or csv_text"}
    except Exception as e:
        return None, {"error": "Data fetch failed", "message": str(e)}

def _build_model(problem_type: str, algorithm: str):
    """Build sklearn model based on problem type and algorithm."""
    models_map = {
        "classification": {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=10),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        },
        "regression": {
            "dummy": DummyRegressor(strategy="mean"),
            "linear": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "decision_tree": DecisionTreeRegressor(random_state=42, max_depth=10),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
    }
    
    if problem_type not in models_map:
        raise ValueError(f"Unknown problem type: {problem_type}")
    if algorithm not in models_map[problem_type]:
        raise ValueError(f"Unknown algorithm '{algorithm}' for {problem_type}")
    
    return models_map[problem_type][algorithm]

def _feature_importance(model, X: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract feature importance from trained model."""
    try:
        # Handle Pipeline models
        if isinstance(model, Pipeline):
            actual_model = model.named_steps.get("model", model)
        else:
            actual_model = model
        
        # Check if model has feature_importances_
        if hasattr(actual_model, "feature_importances_"):
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, "coef_"):
            # For linear models, use absolute coefficients
            coef = actual_model.coef_
            if len(coef.shape) > 1:  # Multi-class
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        else:
            return []
        
        # Create feature importance list
        feature_imp = [
            {"feature": col, "importance": float(imp)}
            for col, imp in zip(X.columns, importances)
        ]
        # Sort by importance
        feature_imp.sort(key=lambda x: x["importance"], reverse=True)
        return feature_imp[:10]  # Top 10 features
    except Exception as e:
        print(f"Feature importance extraction error: {e}")
        return []

def _safe_problem_type(y: pd.Series, user_type: str) -> str:
    """Auto-detect problem type if set to 'auto'."""
    if user_type != "auto":
        return user_type
    
    # Check if target is numeric and has many unique values
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = len(y.unique()) / len(y)
        if unique_ratio > 0.05:  # More than 5% unique values
            return "regression"
    
    return "classification"

def _safe_cv(problem_type: str, y: pd.Series, n_samples: int) -> tuple:
    """Setup appropriate cross-validation strategy."""
    n_splits = min(5, n_samples // 2)
    
    if n_splits < 2:
        return None, "Dataset too small for cross-validation"
    
    if problem_type == "classification":
        # Check if we have enough samples per class
        min_class_count = y.value_counts().min()
        if min_class_count < n_splits:
            n_splits = max(2, min_class_count)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None
    else:
        # For regression, use ShuffleSplit for more robust validation
        # This randomly shuffles and splits, preventing temporal bias
        return ShuffleSplit(n_splits=5, test_size=0.2, random_state=42), None

# ==================== PARALLEL TRAINING HELPER ====================

def train_single_model(algo, problem_type, X, y, cv):
    """
    Train a single machine learning model and return performance metrics.
    
    This function trains one specific algorithm with cross-validation and
    calculates comprehensive metrics for model evaluation.
    
    Args:
        algo (str): Algorithm name ('random_forest', 'linear', etc.)
        problem_type (str): 'classification' or 'regression'
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        cv: Cross-validation splitter object
        
    Returns:
        dict: Model results containing:
            - modelId: Unique identifier
            - algorithm: Algorithm name
            - status: 'ok' or 'error'
            - metrics: Performance metrics (R², MSE, Accuracy, etc.)
            - durationSec: Training time in seconds
            - featureImportance: Top feature importances
            - model_obj: Trained scikit-learn model object
            
    Metrics for Regression:
        - cv_r2_mean: Cross-validated R² score (mean)
        - cv_r2_std: Cross-validated R² score (std dev)
        - train_r2: Training set R² score
        - train_mae: Mean Absolute Error
        - train_mse: Mean Squared Error
        - train_rmse: Root Mean Squared Error
        
    Metrics for Classification:
        - cv_accuracy_mean: Cross-validated accuracy (mean)
        - cv_accuracy_std: Cross-validated accuracy (std dev)
        - train_accuracy: Training set accuracy
        - train_f1: F1 score (weighted)
    """
    model_id = str(uuid.uuid4())
    t0 = time.time()
    try:
        model = _build_model(problem_type, algo)
        metrics = {}
        
        # Cross Validation
        if cv is not None:
            scoring = "accuracy" if problem_type == "classification" else "r2"
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            metrics[f"cv_{scoring}_mean"] = float(cv_scores.mean())
            metrics[f"cv_{scoring}_std"] = float(cv_scores.std())
        
        # Final Fit
        model.fit(X, y)
        preds = model.predict(X)
        
        # Calculate additional metrics
        if problem_type == "classification":
            metrics["train_accuracy"] = float(accuracy_score(y, preds))
            metrics["train_f1"] = float(f1_score(y, preds, average="weighted"))
        else:
            metrics["train_r2"] = float(r2_score(y, preds))
            metrics["train_mae"] = float(mean_absolute_error(y, preds))
            metrics["train_mse"] = float(mean_squared_error(y, preds))
            metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y, preds)))
        
        # Feature Importance
        fi = _feature_importance(model, X)
        
        duration = time.time() - t0
        
        return {
            "modelId": model_id,
            "algorithm": algo,
            "status": "ok",
            "metrics": metrics,
            "durationSec": duration,
            "featureImportance": fi,
            "model_obj": model  # Temporarily hold to store in global dict
        }
    except Exception as e:
        return {"algorithm": algo, "status": "error", "message": str(e)}

# ==================== API ENDPOINTS ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    mongo_status = "connected" if db is not None else "disconnected"
    return {
        "status": "healthy",
        "mongodb": mongo_status,
        "models_in_memory": len(MODELS)
    }

@app.post("/api/train")
async def train(req: TrainRequest):
    """
    Train multiple ML models with advanced text processing and leakage prevention.
    
    This endpoint performs end-to-end automated machine learning including:
    1. Data ingestion and validation
    2. Automatic data leakage detection and removal
    3. Text feature extraction using TF-IDF
    4. Parallel model training with cross-validation
    5. Model persistence and metric calculation
    
    Args:
        req (TrainRequest): Training request containing:
            - csv_text or file_url: Input data
            - target_column: Column to predict
            - feature_columns: Optional list of features
            - algorithm: Specific algorithm or 'auto'
            - problem_type: 'classification', 'regression', or 'auto'
            
    Returns:
        dict: Training results with leaderboard, best model, metrics,
              visualizations data, and data quality warnings
              
    Scientific Features:
        - **Data Leakage Prevention**: Automatically removes ID, date, and
          temporal columns that would artificially inflate accuracy
        - **Text Processing**: TF-IDF vectorization with bigrams (1,2)
          captures phrases like "Special Effects", "Classic Cinema"
        - **Robust Validation**: ShuffleSplit for regression prevents
          temporal bias in cross-validation
        - **Transparency**: Returns residual statistics and warnings
          for low predictive power
    """
    start_total = time.time()
    
    # 1. Data Ingestion
    csv_data, err = _get_csv_text(req.file_url, req.csv_text)
    if err:
        return err
    
    target = req.target or req.target_column
    if not csv_data or not target:
        return {"error": "Missing input", "message": "Target and Data are required."}

    try:
        df = pd.read_csv(StringIO(csv_data))
    except Exception as e:
        return {"error": "CSV parsing failed", "message": str(e)}
    
    if target not in df.columns:
        return {"error": "Column not found", "target": target, "available_columns": df.columns.tolist()}

    # 2. Advanced Preprocessing with Text Support
    X_raw = df[req.feature_columns] if req.feature_columns else df.drop(columns=[target])
    y = df[target]
    
    # ============================================================================
    # CRITICAL: DATA LEAKAGE PREVENTION
    # ============================================================================
    # Data leakage occurs when features contain information about the target
    # that wouldn't be available at prediction time. Common sources:
    #
    # 1. ID Columns: 'user_id', 'transaction_id' - Often correlate with time
    # 2. Date Columns: 'created_at', 'date_added' - Directly encode temporal info
    # 3. Year Columns: 'release_year' when predicting year - Perfect correlation
    #
    # Why This Matters:
    # - Model appears to have 85% accuracy but is actually memorizing IDs
    # - Fails completely on new, unseen data
    # - Violates fundamental ML principle: features must be available at inference
    #
    # Solution: Automatically detect and remove these columns
    # ============================================================================
    leakage_columns = []
    for col in X_raw.columns:
        col_lower = col.lower()
        # Remove ID columns, date columns, and anything with "year" in it
        if any(keyword in col_lower for keyword in ['id', '_id', 'date', 'year', 'added', 'created', 'updated']):
            leakage_columns.append(col)
    
    if leakage_columns:
        print(f"⚠️ Removing potential data leakage columns: {leakage_columns}")
        X_raw = X_raw.drop(columns=leakage_columns)
    
    # ============================================================================
    # TEXT FEATURE EXTRACTION WITH TF-IDF
    # ============================================================================
    # Identify text and numeric columns
    text_columns = []
    numeric_columns = []
    
    for col in X_raw.columns:
        if X_raw[col].dtype == 'object' or X_raw[col].dtype.name == 'string':
            # Check if it looks like text (longer strings, not just categories)
            avg_length = X_raw[col].astype(str).str.len().mean()
            if avg_length > 20:  # Likely text description
                text_columns.append(col)
            else:
                # Categorical - will be one-hot encoded
                numeric_columns.append(col)
        else:
            numeric_columns.append(col)
    
    # Process features with advanced text handling
    if text_columns:
        # Combine all text columns (description, listed_in, etc.)
        X_raw['combined_text'] = X_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # ========================================================================
        # TF-IDF VECTORIZATION (Term Frequency - Inverse Document Frequency)
        # ========================================================================
        # Converts text into numerical features that ML models can process
        #
        # Parameters Explained:
        # - max_features=200: Extract top 200 most important words/phrases
        #   (Increased from 100 for richer representation)
        #
        # - ngram_range=(1, 2): Include both single words AND two-word phrases
        #   Examples: "high" (unigram), "high school" (bigram)
        #   Captures: "special effects", "classic cinema", "crime drama"
        #
        # - stop_words='english': Remove common words (the, is, and, etc.)
        #   Focuses on meaningful content words
        #
        # - min_df=1: Word must appear in at least 1 document
        #   Keeps all words (small datasets benefit from this)
        #
        # - max_df=0.95: Ignore words appearing in >95% of documents
        #   Filters out overly common words that don't discriminate
        #
        # Why TF-IDF?
        # - Weighs words by importance (rare words get higher scores)
        # - Normalizes for document length
        # - Industry standard for text classification/regression
        # ========================================================================
        vectorizer = TfidfVectorizer(
            max_features=200,      # Top 200 features (balance between detail and noise)
            stop_words='english',  # Remove common English words
            ngram_range=(1, 2),    # Include unigrams and bigrams for phrases
            min_df=1,              # Minimum document frequency
            max_df=0.95            # Maximum document frequency (filter common words)
        )
        text_features = vectorizer.fit_transform(X_raw['combined_text']).toarray()
        text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
        text_df = pd.DataFrame(text_features, columns=text_feature_names, index=X_raw.index)
        
        # Process numeric columns
        if numeric_columns:
            numeric_df = X_raw[numeric_columns].copy()
            numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True)).fillna(0)
            numeric_df = pd.get_dummies(numeric_df, drop_first=True)
            X = pd.concat([numeric_df, text_df], axis=1)
        else:
            X = text_df
    else:
        # No text columns - standard preprocessing
        X = X_raw.fillna(X_raw.mean(numeric_only=True)).fillna(0)
        X = pd.get_dummies(X, drop_first=True)
    
    problem_type = _safe_problem_type(y, req.problem_type)
    cv, cv_warning = _safe_cv(problem_type, y, len(df))

    # 3. Parallel Execution
    candidates = ["dummy", "logistic", "decision_tree", "random_forest", "gradient_boosting"] \
        if problem_type == "classification" else \
        ["dummy", "linear", "decision_tree", "random_forest", "gradient_boosting"]
    
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
                model_obj = res.pop("model_obj")
                model_serialized = base64.b64encode(pickle.dumps(model_obj)).decode('utf-8')
                
                MODELS[res["modelId"]] = {
                    "model": model_obj,
                    "columns": X.columns.tolist(),
                    "problemType": problem_type,
                    "algorithm": res["algorithm"],
                    "createdAt": datetime.utcnow().isoformat()
                }
                
                # Store in MongoDB
                if db is not None:
                    try:
                        models_collection.insert_one({
                            "modelId": res["modelId"],
                            "algorithm": res["algorithm"],
                            "problemType": problem_type,
                            "metrics": res["metrics"],
                            "featureImportance": res["featureImportance"],
                            "columns": X.columns.tolist(),
                            "modelData": model_serialized,
                            "createdAt": datetime.utcnow()
                        })
                    except Exception as e:
                        print(f"MongoDB insert error: {e}")
                
            leaderboard.append(res)

    # 4. Final Best Model Selection
    ok_models = [m for m in leaderboard if m["status"] == "ok"]
    if not ok_models:
        return {"error": "Training failed for all models", "details": leaderboard}
    
    metric_key = "cv_accuracy_mean" if problem_type == "classification" else "cv_r2_mean"
    best_model = max(ok_models, key=lambda x: x["metrics"].get(metric_key, -999))
    
    # Add residuals for regression
    residuals = None
    predictions_vs_actual = None
    residual_stats = None
    if problem_type == "regression":
        best_model_obj = MODELS[best_model["modelId"]]["model"]
        y_pred = best_model_obj.predict(X)
        residuals_array = y - y_pred
        residuals = residuals_array.tolist()
        predictions_vs_actual = {
            "actual": y.tolist(),
            "predicted": y_pred.tolist()
        }
        
        # Calculate residual statistics for model quality assessment
        residual_mean = float(residuals_array.mean())
        residual_std = float(residuals_array.std())
        residual_stats = {
            "mean": residual_mean,
            "std": residual_std,
            "mean_abs": float(np.abs(residuals_array).mean()),
            "predictive_power": "Good" if abs(residual_mean) < residual_std * 0.1 else "Low"
        }
    
    # Store training history
    if db is not None:
        try:
            training_history_collection.insert_one({
                "timestamp": datetime.utcnow(),
                "problemType": problem_type,
                "bestModelId": best_model["modelId"],
                "bestAlgorithm": best_model["algorithm"],
                "numSamples": len(df),
                "numFeatures": len(X.columns),
                "targetColumn": target,
                "totalTime": time.time() - start_total,
                "textColumns": text_columns,
                "numericColumns": numeric_columns
            })
        except Exception as e:
            print(f"MongoDB history insert error: {e}")

    return {
        "status": "success",
        "problemType": problem_type,
        "bestModel": best_model,
        "leaderboard": leaderboard,
        "totalTime": time.time() - start_total,
        "dataInfo": {
            "numSamples": len(df),
            "numFeatures": len(X.columns),
            "targetColumn": target,
            "columns": X.columns.tolist(),
            "textColumns": text_columns,
            "numericColumns": numeric_columns,
            "removedLeakageColumns": leakage_columns
        },
        "residuals": residuals,
        "predictionsVsActual": predictions_vs_actual,
        "residualStats": residual_stats
    }

@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Make predictions using a trained model."""
    if req.model_id not in MODELS:
        # Try to load from MongoDB
        if db is not None:
            try:
                model_doc = models_collection.find_one({"modelId": req.model_id})
                if model_doc:
                    model_obj = pickle.loads(base64.b64decode(model_doc["modelData"]))
                    MODELS[req.model_id] = {
                        "model": model_obj,
                        "columns": model_doc["columns"],
                        "problemType": model_doc["problemType"],
                        "algorithm": model_doc["algorithm"]
                    }
                else:
                    raise HTTPException(status_code=404, detail="Model not found")
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = MODELS[req.model_id]
    model = model_info["model"]
    expected_columns = model_info["columns"]
    
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame(req.data)
        
        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only the expected columns in the correct order
        input_df = input_df[expected_columns]
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Get probabilities for classification
        probabilities = None
        if model_info["problemType"] == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df).tolist()
        
        return {
            "status": "success",
            "modelId": req.model_id,
            "predictions": predictions.tolist(),
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/api/models")
async def list_models():
    """List all available models."""
    models_list = []
    
    for model_id, info in MODELS.items():
        models_list.append({
            "modelId": model_id,
            "algorithm": info.get("algorithm", "unknown"),
            "problemType": info.get("problemType", "unknown"),
            "createdAt": info.get("createdAt", "unknown")
        })
    
    return {
        "status": "success",
        "models": models_list,
        "count": len(models_list)
    }

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a specific model."""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Delete from memory
    del MODELS[model_id]
    
    # Delete from MongoDB
    if db is not None:
        try:
            models_collection.delete_one({"modelId": model_id})
        except Exception as e:
            print(f"MongoDB delete error: {e}")
    
    return {
        "status": "success",
        "message": f"Model {model_id} deleted"
    }

@app.get("/api/models/{model_id}/download")
async def download_model(model_id: str):
    """
    Download a trained model as a .pkl file.
    
    This endpoint exports the trained scikit-learn model in pickle format,
    allowing users to:
    - Deploy models in production environments
    - Share models with team members
    - Archive models for future use
    - Load models in custom Python scripts
    
    Args:
        model_id: Unique identifier of the trained model
        
    Returns:
        StreamingResponse: Binary .pkl file download
        
    Usage Example (Python):
        ```python
        import pickle
        import requests
        
        # Download model
        response = requests.get(f"http://localhost:8001/api/models/{model_id}/download")
        with open("model.pkl", "wb") as f:
            f.write(response.content)
        
        # Load and use model
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        predictions = model.predict(X_new)
        ```
    """
    if model_id not in MODELS:
        # Try to load from MongoDB
        if db is not None:
            try:
                model_doc = models_collection.find_one({"modelId": model_id})
                if model_doc:
                    model_obj = pickle.loads(base64.b64decode(model_doc["modelData"]))
                    MODELS[model_id] = {
                        "model": model_obj,
                        "columns": model_doc["columns"],
                        "problemType": model_doc["problemType"],
                        "algorithm": model_doc["algorithm"]
                    }
                else:
                    raise HTTPException(status_code=404, detail="Model not found")
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = MODELS[model_id]
    model = model_info["model"]
    algorithm = model_info.get("algorithm", "model")
    
    # Serialize model to bytes
    model_bytes = pickle.dumps(model)
    
    # Create BytesIO object
    bytes_io = BytesIO(model_bytes)
    
    # Return as downloadable file
    return StreamingResponse(
        bytes_io,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={algorithm}_{model_id[:8]}.pkl"
        }
    )

@app.get("/api/columns")
async def get_columns(file_url: Optional[str] = None):
    """Get column names from a CSV file."""
    if not file_url:
        raise HTTPException(status_code=400, detail="file_url is required")
    
    csv_data, err = _get_csv_text(file_url, None)
    if err:
        raise HTTPException(status_code=400, detail=err.get("message", "Failed to fetch CSV"))
    
    try:
        df = pd.read_csv(StringIO(csv_data))
        return {
            "status": "success",
            "columns": df.columns.tolist(),
            "sample": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
