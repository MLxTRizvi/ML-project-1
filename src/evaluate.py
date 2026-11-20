"""
Model Evaluation and Metrics Computation Module.
================================================
Calculates comprehensive, leakage-resistant classification metrics:
- Accuracy, Precision, Recall, F1 (Fake & Genuine)
- Macro F1, Weighted F1
- ROC-AUC (Receiver Operating Characteristic)
- PR-AUC (Precision-Recall / Average Precision)
- Confusion Matrix & Classification Report
- Benchmarking: Training latency, Inference latency, Serialized Model Size
"""

import os
import time
import tempfile
import joblib
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def evaluate_classifier(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str = "Classifier",
    train_time_sec: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluates a trained model on a held-out test set and reports all metrics.
    
    Args:
        model: Fitted scikit-learn model or Pipeline.
        X_test: Test features (text Series or matrix).
        y_test: Ground-truth binary labels (0 or 1).
        model_name: Human-readable name.
        train_time_sec: Elapsed training time in seconds.
        
    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    # Measure inference latency (predict on test set)
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time_sec = time.perf_counter() - start_pred
    pred_latency_ms_per_100 = (pred_time_sec / len(y_test)) * 100.0 * 1000.0

    # Decision scores / probabilities for ROC-AUC and PR-AUC
    y_scores = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            y_scores = proba[:, 1]
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            y_scores = model.decision_function(X_test)
        except Exception:
            pass

    # Core classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    prec_fake = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_fake = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    prec_honest = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_honest = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_honest = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    roc_auc = float("nan")
    pr_auc = float("nan")
    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_scores)
            pr_auc = average_precision_score(y_test, y_scores)
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Measure serialized model size on disk
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(model, tmp_path, compress=3)
        model_size_mb = os.path.getsize(tmp_path) / (1024.0 * 1024.0)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "model_name": model_name,
        "accuracy": float(round(accuracy, 4)),
        "precision_fake": float(round(prec_fake, 4)),
        "recall_fake": float(round(rec_fake, 4)),
        "f1_fake": float(round(f1_fake, 4)),
        "precision_honest": float(round(prec_honest, 4)),
        "recall_honest": float(round(rec_honest, 4)),
        "f1_honest": float(round(f1_honest, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "roc_auc": float(round(roc_auc, 4)) if not np.isnan(roc_auc) else "N/A",
        "pr_auc": float(round(pr_auc, 4)) if not np.isnan(pr_auc) else "N/A",
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "train_time_sec": float(round(train_time_sec, 2)),
        "pred_latency_ms_per_100": float(round(pred_latency_ms_per_100, 2)),
        "model_size_mb": float(round(model_size_mb, 2)),
        "y_pred": y_pred,
        "y_scores": y_scores
    }


def compute_threshold_metrics(
    y_test: Any,
    y_scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Calculates precision, recall, and F1 across a range of decision thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    records = []
    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        acc = accuracy_score(y_test, preds)
        p = precision_score(y_test, preds, pos_label=1, zero_division=0)
        r = recall_score(y_test, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        records.append({
            "threshold": round(th, 2),
            "accuracy": round(acc, 4),
            "precision_fake": round(p, 4),
            "recall_fake": round(r, 4),
            "f1_fake": round(f1, 4),
            "macro_f1": round(macro, 4)
        })

    return pd.DataFrame(records)
