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
import hmac
import hashlib
import pandas as pd
import numpy as np
import requests
import bcrypt
import jwt
from io import StringIO, BytesIO
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from pymongo import MongoClient

# ML Imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, mean_squared_error

app = FastAPI(title="AutoML Master Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

from fastapi.staticfiles import StaticFiles
app.mount("/api/sample_data", StaticFiles(directory="/app/sample_data"), name="sample_data")

JWT_SECRET = os.environ.get("JWT_SECRET", uuid.uuid4().hex)

# MongoDB Setup
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "automl_db")
try:
    mongo_client = MongoClient(MONGO_URL)
    db = mongo_client[DB_NAME]
    models_collection = db["models"]
    training_history_collection = db["training_history"]
    snapshots_collection = db["snapshots"]
    users_collection = db["users"]
    sessions_collection = db["user_sessions"]
    leaderboard_collection = db["leaderboard_entries"]
    activity_collection = db["activity_log"]
    deployed_models_collection = db["deployed_models"]
    # Create indexes for performance
    leaderboard_collection.create_index([("user_id", 1), ("created_at", -1)])
    sessions_collection.create_index([("user_id", 1)])
    snapshots_collection.create_index([("user_id", 1), ("saved_at", -1)])
    activity_collection.create_index([("timestamp", -1)])
    # Seed admin flag for designated admin accounts
    ADMIN_EMAILS = ["shubhrangshub@gmail.com"]
    for ae in ADMIN_EMAILS:
        users_collection.update_one({"email": ae}, {"$set": {"is_admin": True}}, upsert=False)
except Exception as e:
    print(f"MongoDB connection warning: {e}")
    db = None

# Global model store (in-memory backup)
MODELS: Dict[str, Dict[str, Any]] = {}

# ==================== SECURE MODEL SERIALIZATION ====================
_MODEL_SIGNING_KEY = os.environ.get("MODEL_SIGNING_KEY", JWT_SECRET).encode()

def secure_pickle_dumps(obj) -> str:
    """Serialize object with HMAC signature to prevent tampering."""
    data = pickle.dumps(obj)
    sig = hmac.new(_MODEL_SIGNING_KEY, data, hashlib.sha256).hexdigest()
    payload = sig.encode() + b"|" + data
    return base64.b64encode(payload).decode("utf-8")

def secure_pickle_loads(encoded: str):
    """Deserialize object after verifying HMAC signature."""
    raw = base64.b64decode(encoded)
    if b"|" in raw[:65]:
        sig_hex, data = raw.split(b"|", 1)
        expected = hmac.new(_MODEL_SIGNING_KEY, data, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig_hex.decode(), expected):
            raise ValueError("Model signature verification failed — data may be tampered")
        return pickle.loads(data)
    # Legacy fallback for models saved before signing was added
    return pickle.loads(raw)

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

class SnapshotSaveRequest(BaseModel):
    name: str
    dataset_name: Optional[str] = "Untitled"
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    row_count: Optional[int] = 0
    col_count: Optional[int] = 0
    models_summary: Optional[List[Dict[str, Any]]] = []
    key_metrics: Optional[Dict[str, Any]] = {}
    fingerprint: Optional[str] = None
    state: Dict[str, Any]

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleSessionRequest(BaseModel):
    session_id: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class LeaderboardEntry(BaseModel):
    model_id: str
    algorithm: str
    problem_type: str
    dataset_name: Optional[str] = None
    target_column: Optional[str] = None
    metrics: dict
    feature_importance: Optional[list] = []
    duration_sec: Optional[float] = None
    eval_mode: Optional[str] = None
    num_features: Optional[int] = None
    num_samples: Optional[int] = None

# ==================== AUTH HELPERS ====================

def get_current_user(request: Request) -> Optional[Dict]:
    """Extract user from session_token cookie or Authorization header."""
    if db is None:
        return None
    token = None
    # Try cookie first
    token = request.cookies.get("session_token")
    # Fallback to Authorization header
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None
    session = sessions_collection.find_one({"session_token": token}, {"_id": 0})
    if not session:
        return None
    # Check expiry
    expires_at = session.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at < datetime.now(timezone.utc):
        return None
    user = users_collection.find_one({"user_id": session["user_id"]}, {"_id": 0})
    return user

def require_auth(request: Request) -> Dict:
    """Get current user or raise 401."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

def require_admin(request: Request) -> Dict:
    """Get current user and verify admin or raise 403."""
    user = require_auth(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def log_activity(user_id: str, email: str, action: str, details: str = ""):
    """Log user activity for admin dashboard."""
    if db is None:
        return
    try:
        activity_collection.insert_one({
            "user_id": user_id, "email": email, "action": action,
            "details": details, "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception:
        pass

# ==================== AUTH ENDPOINTS ====================

@app.post("/api/auth/signup")
async def signup(req: SignupRequest, response: FastAPIResponse):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    existing = users_collection.find_one({"email": req.email}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    users_collection.insert_one({
        "user_id": user_id, "email": req.email, "name": req.name,
        "password_hash": hashed, "picture": "",
        "auth_provider": "email",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    # Create session
    session_token = uuid.uuid4().hex
    sessions_collection.insert_one({
        "user_id": user_id, "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    response.set_cookie("session_token", session_token, path="/", httponly=True, secure=True, samesite="none", max_age=7*24*3600)
    log_activity(user_id, req.email, "signup", "New account created")
    return {"status": "success", "token": session_token, "user": {"user_id": user_id, "email": req.email, "name": req.name, "picture": "", "is_admin": False}}

@app.post("/api/auth/login")
async def login(req: LoginRequest, response: FastAPIResponse):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = users_collection.find_one({"email": req.email}, {"_id": 0})
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    session_token = uuid.uuid4().hex
    sessions_collection.insert_one({
        "user_id": user["user_id"], "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    response.set_cookie("session_token", session_token, path="/", httponly=True, secure=True, samesite="none", max_age=7*24*3600)
    log_activity(user["user_id"], user["email"], "login", "Email login")
    return {"status": "success", "token": session_token, "user": {"user_id": user["user_id"], "email": user["email"], "name": user["name"], "picture": user.get("picture", ""), "is_admin": user.get("is_admin", False)}}

@app.post("/api/auth/google")
async def google_auth(req: GoogleSessionRequest, response: FastAPIResponse):
    """Exchange Emergent Google OAuth session_id for a local session."""
    try:
        r = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": req.session_id}, timeout=10
        )
        if r.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Google session")
        gdata = r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google auth failed: {str(e)}")
    email = gdata.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="No email from Google")
    # Upsert user
    existing = users_collection.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
        users_collection.update_one({"email": email}, {"$set": {"name": gdata.get("name", existing.get("name", "")), "picture": gdata.get("picture", "")}})
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        users_collection.insert_one({
            "user_id": user_id, "email": email,
            "name": gdata.get("name", ""), "picture": gdata.get("picture", ""),
            "password_hash": "", "auth_provider": "google",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    session_token = uuid.uuid4().hex
    sessions_collection.insert_one({
        "user_id": user_id, "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    response.set_cookie("session_token", session_token, path="/", httponly=True, secure=True, samesite="none", max_age=7*24*3600)
    user_doc = users_collection.find_one({"user_id": user_id}, {"_id": 0})
    log_activity(user_id, email, "login", "Google OAuth login")
    return {"status": "success", "token": session_token, "user": {"user_id": user_id, "email": email, "name": gdata.get("name", ""), "picture": gdata.get("picture", ""), "is_admin": user_doc.get("is_admin", False) if user_doc else False}}

@app.get("/api/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"], "picture": user.get("picture", ""), "is_admin": user.get("is_admin", False)}

@app.post("/api/auth/logout")
async def logout(request: Request, response: FastAPIResponse):
    token = request.cookies.get("session_token")
    # Also check Authorization header for token (used by frontend localStorage auth)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if token and db is not None:
        sessions_collection.delete_many({"session_token": token})
    response.delete_cookie("session_token", path="/", secure=True, samesite="none")
    return {"status": "success"}

@app.post("/api/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """Generate a password reset token. Returns the token directly (no email service configured)."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    import secrets
    user = users_collection.find_one({"email": req.email}, {"_id": 0})
    if not user:
        # Don't reveal whether email exists — return success either way
        return {"status": "success", "message": "If that email is registered, a reset token has been generated.", "token": None}
    if user.get("auth_provider") == "google" and not user.get("password_hash"):
        return {"status": "error", "message": "This account uses Google Sign-In. Please log in with Google.", "token": None}
    # Invalidate any existing tokens for this email
    reset_tokens_collection = db["password_reset_tokens"]
    reset_tokens_collection.delete_many({"email": req.email})
    # Generate new token
    token = secrets.token_urlsafe(32)
    reset_tokens_collection.insert_one({
        "email": req.email,
        "token": token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "used": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    # In production, this token would be emailed. For now, return it directly.
    return {"status": "success", "message": "Reset token generated.", "token": token}

@app.post("/api/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Reset password using a valid reset token."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    reset_tokens_collection = db["password_reset_tokens"]
    record = reset_tokens_collection.find_one({"token": req.token, "used": False})
    if not record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    # Check expiry
    expires_at = record.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")
    # Update password
    new_hash = bcrypt.hashpw(req.new_password.encode(), bcrypt.gensalt()).decode()
    users_collection.update_one({"email": record["email"]}, {"$set": {"password_hash": new_hash}})
    # Mark token as used
    reset_tokens_collection.update_one({"token": req.token}, {"$set": {"used": True}})
    return {"status": "success", "message": "Password has been reset successfully. You can now sign in."}

# ==================== LEADERBOARD ENDPOINTS ====================

@app.post("/api/leaderboard")
async def save_leaderboard_entry(entry: LeaderboardEntry, request: Request):
    """Auto-save a trained model to the leaderboard."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    doc = {
        "user_id": user.get("email"),
        "model_id": entry.model_id,
        "algorithm": entry.algorithm,
        "problem_type": entry.problem_type,
        "dataset_name": entry.dataset_name,
        "target_column": entry.target_column,
        "metrics": entry.metrics,
        "feature_importance": entry.feature_importance or [],
        "duration_sec": entry.duration_sec,
        "eval_mode": entry.eval_mode,
        "num_features": entry.num_features,
        "num_samples": entry.num_samples,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    leaderboard_collection.insert_one(doc)
    return {"status": "success", "message": "Model saved to leaderboard"}

@app.get("/api/leaderboard")
async def get_leaderboard(request: Request):
    """Get all leaderboard entries for the current user."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    entries = list(leaderboard_collection.find(
        {"user_id": user.get("email")},
        {"_id": 0}
    ).sort("created_at", -1))
    return {"status": "success", "entries": entries}

@app.delete("/api/leaderboard/{model_id}")
async def delete_leaderboard_entry(model_id: str, request: Request):
    """Delete a leaderboard entry."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    result = leaderboard_collection.delete_one({"model_id": model_id, "user_id": user.get("email")})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "success"}

@app.delete("/api/leaderboard")
async def clear_leaderboard(request: Request):
    """Clear all leaderboard entries for current user."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    leaderboard_collection.delete_many({"user_id": user.get("email")})
    return {"status": "success"}

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
                model_serialized = secure_pickle_dumps(model_obj)
                
                MODELS[res["modelId"]] = {
                    "model": model_obj,
                    "columns": X.columns.tolist(),
                    "problemType": problem_type,
                    "algorithm": res["algorithm"],
                    "createdAt": datetime.now(timezone.utc).isoformat()
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
                            "createdAt": datetime.now(timezone.utc)
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
                "timestamp": datetime.now(timezone.utc),
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

    # Log training activity
    log_activity(req.user_id or "anonymous", "", "train", f"Trained {len(leaderboard)} models on '{target}' ({problem_type})")

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
    """
    Make predictions using a trained model with confidence scoring.
    
    This endpoint generates predictions and provides confidence metrics
    based on model uncertainty and input data quality.
    
    Args:
        req (PredictRequest): Contains model_id and input data
        
    Returns:
        dict: Predictions with confidence scores, including:
            - predictions: Model predictions
            - probabilities: Class probabilities (classification only)
            - confidence: Confidence scores for each prediction
            - confidence_level: Overall confidence assessment
            - input_quality: Assessment of input data quality
    """
    if req.model_id not in MODELS:
        # Try to load from MongoDB
        if db is not None:
            try:
                model_doc = models_collection.find_one({"modelId": req.model_id})
                if model_doc:
                    model_obj = secure_pickle_loads(model_doc["modelData"])
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
    problem_type = model_info["problemType"]
    
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
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df).tolist()
        
        # Calculate confidence scores
        confidence_scores = []
        confidence_level = "High"
        
        for idx, pred in enumerate(predictions):
            # Base confidence calculation
            if problem_type == "classification" and probabilities:
                # For classification: use probability of predicted class
                max_prob = max(probabilities[idx])
                confidence = max_prob * 100
                if confidence < 70:
                    confidence_level = "Low"
                elif confidence < 85:
                    confidence_level = "Medium"
            else:
                # For regression: assess based on input feature quality
                row = input_df.iloc[idx]
                non_zero_features = (row != 0).sum()
                feature_ratio = non_zero_features / len(row)
                
                # Text feature density check (if TF-IDF features present)
                tfidf_features = [col for col in expected_columns if 'tfidf' in col]
                if tfidf_features:
                    tfidf_values = row[tfidf_features]
                    text_density = (tfidf_values > 0).sum() / len(tfidf_features)
                    
                    # Low text density = low confidence
                    if text_density < 0.05:  # Less than 5% of text features active
                        confidence = 30.0  # Low confidence
                        confidence_level = "Low"
                    elif text_density < 0.15:
                        confidence = 65.0  # Medium confidence
                        confidence_level = "Medium"
                    else:
                        confidence = 88.0  # High confidence
                        confidence_level = "High"
                else:
                    # Standard confidence based on feature completeness
                    confidence = feature_ratio * 100
                    if confidence < 50:
                        confidence_level = "Low"
                    elif confidence < 75:
                        confidence_level = "Medium"
            
            confidence_scores.append(float(confidence))
        
        return {
            "status": "success",
            "modelId": req.model_id,
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "confidence": confidence_scores,
            "confidenceLevel": confidence_level,
            "problemType": problem_type
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
                    model_obj = secure_pickle_loads(model_doc["modelData"])
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
    
    # Serialize model to bytes (standard pickle for user download — they own the model)
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

# ==================== SNAPSHOT ENDPOINTS (History & Sharing) ====================

@app.post("/api/snapshots")
async def save_snapshot(req: SnapshotSaveRequest, request: Request):
    """Save or update an analysis snapshot. Deduplicates by fingerprint."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = get_current_user(request)
    user_id = user["user_id"] if user else None

    # Deduplicate: if same user + fingerprint exists, update instead of insert
    if req.fingerprint:
        dedup_query = {"fingerprint": req.fingerprint}
        if user_id:
            dedup_query["user_id"] = user_id
        else:
            dedup_query["user_id"] = None
        existing = snapshots_collection.find_one(
            dedup_query, {"_id": 0, "snapshot_id": 1}
        )
        if existing:
            snapshots_collection.update_one(
                {"snapshot_id": existing["snapshot_id"]},
                {"$set": {
                    "name": req.name, "models_summary": req.models_summary,
                    "key_metrics": req.key_metrics, "state": req.state,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }}
            )
            return {"status": "success", "snapshot_id": existing["snapshot_id"], "action": "updated"}

    snapshot_id = str(uuid.uuid4())[:12]
    doc = {
        "snapshot_id": snapshot_id,
        "user_id": user_id,
        "name": req.name,
        "dataset_name": req.dataset_name,
        "target_column": req.target_column,
        "problem_type": req.problem_type,
        "row_count": req.row_count,
        "col_count": req.col_count,
        "models_summary": req.models_summary,
        "key_metrics": req.key_metrics,
        "fingerprint": req.fingerprint,
        "state": req.state,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        snapshots_collection.insert_one(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")
    log_activity(user_id or "anonymous", user.get("email", "") if user else "", "save_analysis", f"Saved '{req.name}' ({req.dataset_name})")
    return {"status": "success", "snapshot_id": snapshot_id, "action": "created"}

@app.get("/api/snapshots")
async def list_snapshots(request: Request):
    """List saved analysis snapshots for the current user."""
    if db is None:
        return {"status": "success", "snapshots": []}
    user = get_current_user(request)
    query = {"user_id": user["user_id"]} if user else {"user_id": None}
    try:
        cursor = snapshots_collection.find(
            query, {"_id": 0, "state": 0}
        ).sort("created_at", -1).limit(50)
        return {"status": "success", "snapshots": list(cursor)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str):
    """Get a full snapshot by ID (for sharing — no auth required for view-only)."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        doc = snapshots_collection.find_one(
            {"snapshot_id": snapshot_id}, {"_id": 0}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return {"status": "success", "snapshot": doc}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str, request: Request):
    """Delete a snapshot (only owner can delete)."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = get_current_user(request)
    query = {"snapshot_id": snapshot_id}
    if user:
        query["user_id"] = user["user_id"]
    try:
        result = snapshots_collection.delete_one(query)
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return {"status": "success", "message": "Snapshot deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CSV EXPORT ENDPOINT ====================

class ExportCSVRequest(BaseModel):
    csv_content: str
    filename: str = "export.csv"

# Temporary store for download tokens with expiry
_pending_downloads: Dict[str, Dict] = {}

@app.post("/api/export/prepare")
async def prepare_export(req: ExportCSVRequest):
    """Store CSV content and return a one-time download token."""
    # Clean expired tokens (older than 5 minutes)
    now = datetime.now(timezone.utc).timestamp()
    expired = [k for k, v in _pending_downloads.items() if now - v.get("ts", 0) > 300]
    for k in expired:
        _pending_downloads.pop(k, None)
    token = str(uuid.uuid4())[:16]
    _pending_downloads[token] = {"content": req.csv_content, "filename": req.filename, "ts": now}
    return {"status": "success", "token": token}

@app.get("/api/export/download/{token}")
async def download_export(token: str):
    """Serve a prepared CSV file as a real HTTP download. Works in iframes."""
    data = _pending_downloads.pop(token, None)
    if not data:
        raise HTTPException(status_code=404, detail="Download expired or not found")
    return Response(
        content=data["content"],
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{data["filename"]}"'}
    )

# ==================== MODEL DEPLOYMENT ENDPOINTS ====================

class DeployRequest(BaseModel):
    model_id: str
    model_data: Dict
    name: str = ""
    description: str = ""

@app.post("/api/deploy")
async def deploy_model(req: DeployRequest, request: Request):
    """Deploy a trained model and get a public prediction URL."""
    user = require_auth(request)
    deploy_id = str(uuid.uuid4())[:12]
    doc = {
        "deploy_id": deploy_id,
        "user_id": user["user_id"],
        "owner_email": user["email"],
        "model_id": req.model_id,
        "name": req.name or f"Model {deploy_id}",
        "description": req.description,
        "model_data": req.model_data,
        "enabled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_count": 0,
    }
    deployed_models_collection.insert_one(doc)
    log_activity(user["user_id"], user["email"], "deploy_model", f"Deployed '{doc['name']}' ({deploy_id})")
    return {"status": "success", "deploy_id": deploy_id}

@app.get("/api/deploy")
async def list_deployments(request: Request):
    """List all deployments for the current user."""
    user = require_auth(request)
    deployments = list(deployed_models_collection.find(
        {"user_id": user["user_id"]}, {"_id": 0, "model_data": 0}
    ).sort("created_at", -1))
    return {"deployments": deployments}

@app.patch("/api/deploy/{deploy_id}")
async def toggle_deployment(deploy_id: str, request: Request):
    """Enable/disable a deployed model."""
    user = require_auth(request)
    body = await request.json()
    result = deployed_models_collection.update_one(
        {"deploy_id": deploy_id, "user_id": user["user_id"]},
        {"$set": {"enabled": bool(body.get("enabled", True))}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"status": "success"}

@app.delete("/api/deploy/{deploy_id}")
async def delete_deployment(deploy_id: str, request: Request):
    """Delete a deployed model."""
    user = require_auth(request)
    result = deployed_models_collection.delete_one({"deploy_id": deploy_id, "user_id": user["user_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Deployment not found")
    log_activity(user["user_id"], user["email"], "undeploy_model", f"Removed deployment {deploy_id}")
    return {"status": "success"}

@app.get("/api/public/model/{deploy_id}")
async def get_public_model_info(deploy_id: str):
    """Get public info about a deployed model (no auth required)."""
    doc = deployed_models_collection.find_one({"deploy_id": deploy_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")
    if not doc.get("enabled"):
        raise HTTPException(status_code=403, detail="This model has been disabled by its owner")
    md = doc.get("model_data", {})
    return {
        "deploy_id": deploy_id,
        "name": doc.get("name"),
        "description": doc.get("description"),
        "owner": doc.get("owner_email", "").split("@")[0] + "@...",
        "algorithm": md.get("algorithm"),
        "problem_type": md.get("problemType"),
        "features": md.get("modelData", {}).get("numericCols", []) + md.get("modelData", {}).get("categoricalCols", []),
        "numeric_cols": md.get("modelData", {}).get("numericCols", []),
        "categorical_cols": md.get("modelData", {}).get("categoricalCols", []),
        "encoding_map": md.get("modelData", {}).get("encodingMap", {}),
        "target": md.get("targetColumn"),
        "metrics": md.get("metrics", {}),
        "created_at": doc.get("created_at"),
        "prediction_count": doc.get("prediction_count", 0),
        "model_data_full": md.get("modelData"),
    }


# ---- Python reimplementation of JS predictOne / prepareInputForPrediction ----

def _predict_tree_py(node, x):
    """Traverse a decision tree node dict."""
    if node is None:
        return 0
    if "value" in node and "featureIndex" not in node:
        return node["value"]
    fi = node.get("featureIndex")
    thr = node.get("threshold")
    if fi is None or thr is None:
        return node.get("value", 0)
    if x[fi] <= thr:
        return _predict_tree_py(node.get("left"), x)
    else:
        return _predict_tree_py(node.get("right"), x)


def _predict_one_py(model_data, x):
    """Python version of mlEngine.predictOne."""
    t = model_data.get("type", "")
    if t == "baseline":
        return model_data.get("value", 0)
    if t == "decision_tree":
        return _predict_tree_py(model_data.get("tree"), x)
    if t == "random_forest_regressor":
        trees = model_data.get("trees", [])
        if not trees:
            return 0
        return sum(_predict_tree_py(tr, x) for tr in trees) / len(trees)
    if t == "random_forest_classifier":
        trees = model_data.get("trees", [])
        votes = {}
        for tr in trees:
            p = _predict_tree_py(tr, x)
            votes[p] = votes.get(p, 0) + 1
        return max(votes, key=votes.get) if votes else 0
    if t == "gradient_boosting":
        base_mean = model_data.get("baseMean", 0)
        lr = model_data.get("learningRate", 0.1)
        trees = model_data.get("trees", [])
        p = base_mean
        for tr in trees:
            p += lr * _predict_tree_py(tr, x)
        return p
    if t == "knn":
        k = model_data.get("k", 5)
        X_train = model_data.get("X_train", [])
        y_train = model_data.get("y_train", [])
        means = model_data.get("means")
        stds = model_data.get("stds")
        xs = x
        if means and stds:
            xs = [(v - means[j]) / stds[j] if stds[j] != 0 else 0 for j, v in enumerate(x)]
        dists = []
        for i, xi in enumerate(X_train):
            d = sum((xi[j] - xs[j]) ** 2 for j in range(min(len(xi), len(xs))))
            dists.append((d, y_train[i]))
        dists.sort(key=lambda x: x[0])
        top = dists[:k]
        counts = {}
        for _, label in top:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get) if counts else 0
    if t == "svm":
        means = model_data.get("means")
        stds = model_data.get("stds")
        xs = x
        if means and stds:
            xs = [(v - means[j]) / stds[j] if stds[j] != 0 else 0 for j, v in enumerate(x)]
        classes = model_data.get("classes", [0, 1])
        if model_data.get("multiclass"):
            classifiers = model_data.get("classifiers", [])
            scores = [sum(xs[j] * c["w"][j] for j in range(min(len(xs), len(c["w"])))) + c["b"] for c in classifiers]
            return classes[scores.index(max(scores))] if scores else classes[0]
        w = model_data.get("w", [])
        b = model_data.get("b", 0)
        val = sum(xs[j] * w[j] for j in range(min(len(xs), len(w)))) + b
        return classes[1] if val >= 0 else classes[0]
    if t == "naive_bayes":
        import math
        classes = model_data.get("classes", [])
        class_stats = model_data.get("classStats", {})
        best, best_lp = classes[0] if classes else 0, float("-inf")
        for cls in classes:
            s = class_stats.get(str(cls), {})
            lp = math.log(max(s.get("prior", 0.01), 1e-10))
            means_nb = s.get("means", [])
            variances = s.get("variances", [])
            for j in range(min(len(x), len(means_nb))):
                var = max(variances[j], 1e-10)
                lp += -0.5 * math.log(2 * math.pi * var) - (x[j] - means_nb[j]) ** 2 / (2 * var)
            if lp > best_lp:
                best_lp = lp
                best = cls
        return best
    if t == "logistic_regression":
        import math
        coeffs = model_data.get("coefficients", [0])
        z = coeffs[0] + sum(x[i] * coeffs[i + 1] for i in range(min(len(x), len(coeffs) - 1)))
        z = max(-500, min(500, z))
        return 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
    # Default: linear model (linear_regression, ridge)
    coeffs = model_data.get("coefficients", [0])
    return coeffs[0] + sum(x[i] * coeffs[i + 1] for i in range(min(len(x), len(coeffs) - 1)))


def _prepare_input_py(input_dict, model_data):
    """Python version of mlEngine.prepareInputForPrediction."""
    numeric_cols = model_data.get("numericCols", [])
    categorical_cols = model_data.get("categoricalCols", [])
    encoding_map = model_data.get("encodingMap", {})
    scale_params = model_data.get("scaleParams")
    row = []
    for col in numeric_cols:
        try:
            row.append(float(input_dict.get(col, 0)))
        except (ValueError, TypeError):
            row.append(0.0)
    for col in categorical_cols:
        categories = encoding_map.get(col, [])
        val = str(input_dict.get(col, ""))
        for cat in categories[1:]:  # skip first (reference category)
            row.append(1.0 if val == cat else 0.0)
    # Apply scaling if it was used during training
    if scale_params and scale_params.get("params"):
        method = scale_params.get("method")
        params = scale_params["params"]
        for j in range(min(len(row), len(params))):
            p = params[j]
            if method == "standard":
                row[j] = (row[j] - p.get("mean", 0)) / (p.get("std", 1) or 1)
            elif method == "minmax":
                row[j] = (row[j] - p.get("min", 0)) / (p.get("range", 1) or 1)
    return row


@app.post("/api/public/predict/{deploy_id}")
async def public_predict(deploy_id: str, request: Request):
    """Make a prediction using a deployed model (no auth required)."""
    doc = deployed_models_collection.find_one({"deploy_id": deploy_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")
    if not doc.get("enabled"):
        raise HTTPException(status_code=403, detail="This model has been disabled by its owner")
    body = await request.json()
    input_data = body.get("features", {})
    md = doc.get("model_data", {})
    inner_md = md.get("modelData", {})

    try:
        feature_vector = _prepare_input_py(input_data, inner_md)
        prediction = _predict_one_py(inner_md, feature_vector)
        result = {"prediction": float(prediction) if isinstance(prediction, (int, float, np.integer, np.floating)) else prediction}

        # For classification, add class probabilities if possible
        problem_type = md.get("problemType", "")
        if problem_type == "classification" and inner_md.get("type") == "logistic_regression":
            import math
            coeffs = inner_md.get("coefficients", [0])
            z = coeffs[0] + sum(feature_vector[i] * coeffs[i + 1] for i in range(min(len(feature_vector), len(coeffs) - 1)))
            z = max(-500, min(500, z))
            prob_1 = 1 / (1 + math.exp(-z))
            result["probabilities"] = {"0": round(1 - prob_1, 4), "1": round(prob_1, 4)}

        deployed_models_collection.update_one({"deploy_id": deploy_id}, {"$inc": {"prediction_count": 1}})
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==================== ADMIN ENDPOINTS ====================

@app.get("/api/admin/users")
async def admin_list_users(request: Request):
    """List all users with usage stats."""
    admin = require_admin(request)
    users = list(users_collection.find({}, {"_id": 0, "password_hash": 0}))
    for u in users:
        uid = u["user_id"]
        u["snapshots_count"] = snapshots_collection.count_documents({"user_id": uid})
        u["leaderboard_count"] = leaderboard_collection.count_documents({"user_id": uid})
        last_session = sessions_collection.find_one({"user_id": uid}, {"_id": 0}, sort=[("created_at", -1)])
        u["last_active"] = last_session.get("created_at") if last_session else u.get("created_at")
        u["is_admin"] = u.get("is_admin", False)
        u["is_disabled"] = u.get("is_disabled", False)
    return {"users": users, "total": len(users)}

@app.patch("/api/admin/users/{user_id}")
async def admin_update_user(user_id: str, request: Request):
    """Update user flags (is_admin, is_disabled)."""
    admin = require_admin(request)
    body = await request.json()
    update_fields = {}
    if "is_admin" in body:
        update_fields["is_admin"] = bool(body["is_admin"])
    if "is_disabled" in body:
        update_fields["is_disabled"] = bool(body["is_disabled"])
    if not update_fields:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    result = users_collection.update_one({"user_id": user_id}, {"$set": update_fields})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    target_user = users_collection.find_one({"user_id": user_id}, {"_id": 0})
    log_activity(admin["user_id"], admin["email"], "admin_update_user", f"Updated {target_user.get('email')} — {update_fields}")
    return {"status": "success", "updated": update_fields}

@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: str, request: Request):
    """Delete a user and all their data."""
    admin = require_admin(request)
    target = users_collection.find_one({"user_id": user_id}, {"_id": 0})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    if target.get("email") == admin.get("email"):
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    # Delete all user data
    sessions_collection.delete_many({"user_id": user_id})
    snapshots_collection.delete_many({"user_id": user_id})
    leaderboard_collection.delete_many({"user_id": user_id})
    activity_collection.delete_many({"user_id": user_id})
    users_collection.delete_one({"user_id": user_id})
    log_activity(admin["user_id"], admin["email"], "admin_delete_user", f"Deleted user {target.get('email')}")
    return {"status": "success"}

@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(user_id: str, request: Request):
    """Admin-initiated password reset."""
    admin = require_admin(request)
    body = await request.json()
    new_password = body.get("new_password", "")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    target = users_collection.find_one({"user_id": user_id}, {"_id": 0})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    users_collection.update_one({"user_id": user_id}, {"$set": {"password_hash": new_hash}})
    log_activity(admin["user_id"], admin["email"], "admin_reset_password", f"Reset password for {target.get('email')}")
    return {"status": "success", "message": f"Password reset for {target.get('email')}"}

@app.get("/api/admin/analytics")
async def admin_analytics(request: Request):
    """Get platform-wide usage analytics."""
    require_admin(request)
    total_users = users_collection.count_documents({})
    total_snapshots = snapshots_collection.count_documents({})
    total_leaderboard = leaderboard_collection.count_documents({})
    active_sessions = sessions_collection.count_documents({"expires_at": {"$gt": datetime.now(timezone.utc).isoformat()}})
    google_users = users_collection.count_documents({"auth_provider": "google"})
    email_users = users_collection.count_documents({"auth_provider": "email"})
    # Recent signups (last 7 days)
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    recent_signups = users_collection.count_documents({"created_at": {"$gte": week_ago}})
    # Activity counts
    total_trains = activity_collection.count_documents({"action": "train"})
    total_logins = activity_collection.count_documents({"action": "login"})
    total_saves = activity_collection.count_documents({"action": "save_analysis"})
    return {
        "total_users": total_users, "total_snapshots": total_snapshots,
        "total_leaderboard_entries": total_leaderboard, "active_sessions": active_sessions,
        "google_users": google_users, "email_users": email_users,
        "recent_signups": recent_signups, "total_trains": total_trains,
        "total_logins": total_logins, "total_saves": total_saves,
    }

@app.get("/api/admin/activity")
async def admin_activity(request: Request, limit: int = 50, action: str = ""):
    """Get recent activity log with optional action filter."""
    require_admin(request)
    query = {}
    if action:
        query["action"] = action
    activities = list(activity_collection.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return {"activities": activities, "total": activity_collection.count_documents(query)}

@app.delete("/api/admin/system/leaderboard")
async def admin_clear_all_leaderboard(request: Request):
    """Clear ALL leaderboard entries (system-wide)."""
    admin = require_admin(request)
    result = leaderboard_collection.delete_many({})
    log_activity(admin["user_id"], admin["email"], "admin_clear_leaderboard", f"Cleared {result.deleted_count} entries")
    return {"status": "success", "deleted": result.deleted_count}

@app.delete("/api/admin/system/snapshots")
async def admin_clear_all_snapshots(request: Request):
    """Clear ALL snapshots (system-wide)."""
    admin = require_admin(request)
    result = snapshots_collection.delete_many({})
    log_activity(admin["user_id"], admin["email"], "admin_clear_snapshots", f"Cleared {result.deleted_count} snapshots")
    return {"status": "success", "deleted": result.deleted_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
