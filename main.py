from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import uuid
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from io import StringIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}

class TrainRequest(BaseModel):
    csv_text: str
    target: str
    problem_type: str = "auto"

@app.post("/train")
def train(req: TrainRequest):
    df = pd.read_csv(StringIO(req.csv_text))
    y = df[req.target]
    X = df.drop(columns=[req.target])
    X = pd.get_dummies(X, drop_first=True)

    problem_type = req.problem_type
    if problem_type == "auto":
        problem_type = "classification" if y.nunique() < 20 else "regression"

    # ===== OPTION C: CROSS-VALIDATION BASED TRAINING =====

if problem_type == "regression":
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LinearRegression())
    ])

    cv_folds = min(5, len(df))
    if cv_folds < 2:
        cv_folds = 2

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    model.fit(X, y)
    preds_full = model.predict(X)

    metrics = {
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "mae_full_fit": float(mean_absolute_error(y, preds_full))
    }

    feature_importance = []
    sample_actual = y.head(30).tolist()
    sample_pred = preds_full[:30].tolist()

else:  # classification
    else:  # classification
    # Choose folds safely
    n = len(df)
n_classes = y.nunique()
min_class_count = y.value_counts().min()

# 🚨 FRIENDLY ERROR MESSAGE (ADD HERE)
if min_class_count < 2:
    return {
        "error": "Not enough samples in one class for cross-validation.",
        "message": "Each class must have at least 2 rows. Add more data or reduce imbalance.",
        "classCounts": y.value_counts().to_dict()
    }

# folds cannot exceed smallest class count (for stratified cv)
cv_folds = min(5, n, min_class_count)
if cv_folds < 2:
    cv_folds = 2

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Model (no scaler needed here for your demo; keeps it robust)
    model = LogisticRegression(max_iter=2000)

    # Cross-val accuracy
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Fit on full data for sample predictions
    model.fit(X, y)
    preds_full = model.predict(X)

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "f1_full_fit": float(f1_score(y, preds_full, average="weighted"))
    }

    feature_importance = []
    sample_actual = y.head(30).tolist()
    sample_pred = preds_full[:30].tolist()

    model_id = str(uuid.uuid4())
    MODELS[model_id] = {"model": model, "columns": X.columns.tolist()}

    fi = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(X.columns, model.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )[:20]

    return {
    "modelId": str(uuid.uuid4()),
    "problemType": problem_type,
    "metrics": metrics,
    "featureImportance": feature_importance,
    "sample": {
        "actual": sample_actual,
        "predicted": sample_pred
    }
}

