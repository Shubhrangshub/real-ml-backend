from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import uuid
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "regression":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "r2": round(r2_score(y_test, preds), 4),
            "mae": round(mean_absolute_error(y_test, preds), 4),
        }
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds, average="weighted"), 4),
        }

    model_id = str(uuid.uuid4())
    MODELS[model_id] = {"model": model, "columns": X.columns.tolist()}

    fi = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(X.columns, model.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )[:20]

    return {
        "modelId": model_id,
        "problemType": problem_type,
        "metrics": metrics,
        "featureImportance": fi,
        "sample": {
            "actual": y_test.head(30).tolist(),
            "predicted": preds[:30].tolist()
        }
    }
