from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from io import StringIO
import uuid

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score


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
    problem_type: str = "auto"   # "auto" | "regression" | "classification"


@app.get("/")
def home():
    return {"status": "ok", "message": "ML backend running", "docs": "/docs"}


@app.post("/train")
def train(req: TrainRequest):
    df = pd.read_csv(StringIO(req.csv_text))

    if req.target not in df.columns:
        return {"error": f"Target column '{req.target}' not found.", "columns": df.columns.tolist()}

    y = df[req.target]
    X = df.drop(columns=[req.target])
    X = pd.get_dummies(X, drop_first=True)

    problem_type = req.problem_type
    if problem_type == "auto":
        problem_type = "classification" if y.nunique() < 20 else "regression"

    # ===== OPTION C: CROSS-VALIDATION BASED TRAINING =====
    if problem_type == "regression":
        model = LinearRegression()

        cv_folds = min(5, len(df))
        if cv_folds < 2:
            cv_folds = 2

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        # Fit on full data for sample predictions
        model.fit(X, y)
        preds_full = model.predict(X)

        metrics = {
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "mae_full_fit": float(mean_absolute_error(y, preds_full))
        }

        feature_importance = []  # (we'll add permutation importance later)
        sample_actual = y.head(30).tolist()
        sample_pred = preds_full[:30].tolist()

    else:  # classification
        # Friendly error: need at least 2 samples in each class for StratifiedKFold
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min())

        if min_class_count < 2:
            return {
                "error": "Not enough samples in one class for cross-validation.",
                "message": "Each class must have at least 2 rows. Add more data or reduce imbalance.",
                "classCounts": class_counts.to_dict()
            }

        model = LogisticRegression(max_iter=2000)

        cv_folds = min(5, len(df), min_class_count)
        if cv_folds < 2:
            cv_folds = 2

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        # Fit on full data for sample predictions
        model.fit(X, y)
        preds_full = model.predict(X)

        metrics = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "f1_full_fit": float(f1_score(y, preds_full, average="weighted"))
        }

        feature_importance = []  # (we'll add permutation importance later)
        sample_actual = y.head(30).tolist()
        sample_pred = preds_full[:30].tolist()

    # store model for later predict endpoint
    model_id = str(uuid.uuid4())
    MODELS[model_id] = {"model": model, "columns": X.columns.tolist()}

    return {
        "modelId": model_id,
        "problemType": problem_type,
        "metrics": metrics,
        "featureImportance": feature_importance,
        "sample": {
            "actual": sample_actual,
            "predicted": sample_pred
        }
    }

