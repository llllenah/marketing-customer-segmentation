"""
Churn prediction model training and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    average_precision_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "is_churned",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Train Logistic Regression, Random Forest, and XGBoost for churn prediction.
    
    Returns dict with models, predictions, and metrics.
    """
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]:,} samples ({y_train.mean()*100:.1f}% churn)")
    print(f"Test:  {X_test.shape[0]:,} samples ({y_test.mean()*100:.1f}% churn)")

    # Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state, eval_metric="logloss",
            verbosity=0,
        )

    results = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        # Use scaled data for LR, raw for tree models
        X_tr = X_train_scaled if "Logistic" in name else X_train
        X_te = X_test_scaled if "Logistic" in name else X_test

        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        print(f"\nROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        X_cv = X_train_scaled if "Logistic" in name else X_train
        cv_scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="roc_auc")
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "roc_auc": roc_auc,
            "ap": ap,
            "cv_scores": cv_scores,
        }

    return results, X_test, y_test, scaler, feature_cols


def plot_model_comparison(results: dict, y_test: pd.Series) -> None:
    """Plot ROC curves and confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))

    colors = ["#3498DB", "#2ECC71", "#E74C3C"]

    # ROC curves
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                     color=colors[i], linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend(loc="lower right")

    # Confusion matrices
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"]).plot(
            ax=axes[i + 1], cmap="Blues", colorbar=False
        )
        axes[i + 1].set_title(f"{name}")

    plt.suptitle("Churn Prediction Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(results: dict, feature_cols: list[str], top_n: int = 15) -> None:
    """Plot feature importance for tree-based models."""
    tree_models = {k: v for k, v in results.items() if k != "Logistic Regression"}

    if not tree_models:
        print("No tree-based models found.")
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, tree_models.items()):
        importances = res["model"].feature_importances_
        idx = np.argsort(importances)[-top_n:]

        ax.barh(
            [feature_cols[i] for i in idx],
            importances[idx],
            color="#3498DB",
            edgecolor="none",
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{name} — Top {top_n} Features")

    plt.suptitle("Feature Importance Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
