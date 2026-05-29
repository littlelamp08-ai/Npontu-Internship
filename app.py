import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- PIPELINE BUILD & TRAINING ---
def build_and_train():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2, random_state=42, stratify=y)
    
    # Encapsulate preprocessing inside the pipeline to prevent data leakage
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Quick test valuation
    preds = pipeline.predict(X_test)
    print(f"Baseline Train Accuracy: {accuracy_score(y_test, preds):.4f}")
    joblib.dump(pipeline, "model_pipeline.pkl")

try:
    model = joblib.load("model_pipeline.pkl")
except FileNotFoundError:
    build_and_train()
    model = joblib.load("model_pipeline.pkl")

# --- REST API DEPLOYMENT ---
app = FastAPI(title="ML Pipeline API")

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        features = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]])
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()
        return {"class_id": pred, "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))