from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional, List, Any
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
   class TrainRequest(BaseModel):
    csv_text: Optional[str] = None
    file_url: Optional[str] = None

    target: Optional[str] = None
    target_column: Optional[str] = None

    problem_type: str = "auto"

    # optional extras from Base-44 (accepted, ignored)
    feature_columns: Optional[List[str]] = None
    algorithm: Optional[str] = None
    model_id: Optional[str] = None
    extra: Optional[Any] = None

@app.post("/train")
@app.post("/train")
def train(req: TrainRequest):
    # 1) Get CSV text
    csv_text = req.csv_text

    if (csv_text is None or csv_text.strip() == "") and req.file_url:
        try:
            r = requests.get(req.file_url, timeout=30)
            r.raise_for_status()
            csv_text = r.text
        except Exception as e:
            return {"error": "Failed to download CSV from file_url", "message": str(e), "file_url": req.file_url}

    if csv_text is None or csv_text.strip() == "":
        return {"error": "Missing csv_text or file_url", "message": "Provide either csv_text or file_url."}

    # 2) Resolve target column name
    target = req.target or req.target_column
    if not target:
        return {"error": "Missing target", "message": "Provide target or target_column."}

    # 3) Load CSV
    df = pd.read_csv(StringIO(csv_text))

    if target not in df.columns:
        return {"error": f"Target column '{target}' not found.", "columns": df.columns.tolist()}

    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)

    problem_type = req.problem_type
    if problem_type == "auto":
        problem_type = "classification" if y.nunique() < 20 else "regression"

    # ... keep the rest of your regression/classification training code as-is ...


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

