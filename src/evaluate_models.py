"""
Sklearn multi-model evaluation + Word report generator (simplified)
- Dataset: Breast Cancer (sklearn)
- Models: Logistic Regression, SVM (RBF), Random Forest, k-NN, Decision Tree, Gaussian NB
- Metrics: Accuracy, F1 (weighted), Confusion Matrix (+ 5-fold CV accuracy)
- Report: Concepts-focused (no full code, no line-by-line; explains ML parts only)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def evaluate_models():
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing (scaling) for numeric features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Define models (keep it simple, no Pipelines)
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), True),
        ("SVM (RBF)", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42), True),
        ("k-NN", KNeighborsClassifier(n_neighbors=5), True),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42), False),
        ("Decision Tree", DecisionTreeClassifier(random_state=42), False),
        ("Gaussian NB", GaussianNB(), False),
    ]

    results = {}

    for name, model, use_scaled in models:
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test

        # Fit and predict
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)

        # Simple 5-fold CV accuracy on full data
        cv_scores = cross_val_score(
            model, (X if not use_scaled else scaler.fit_transform(X)), y, cv=5, scoring="accuracy"
        )

        results[name] = {
            "accuracy": float(acc),
            "f1_weighted": float(f1w),
            "cv_mean_acc": float(np.mean(cv_scores)),
            "cv_std_acc": float(np.std(cv_scores)),
            "confusion_matrix": cm,
        }

    results["__dataset__"] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_names": list(load_breast_cancer().target_names),
    }

    return results


def main():
    print("Training and evaluating models on sklearn Breast Cancer dataset ...")
    results = evaluate_models()

    # Console summary
    for name, info in results.items():
        if name == "__dataset__":
            continue
        print("-" * 60)
        print(name)
        print(f"Accuracy: {info['accuracy']:.4f}")
        print(f"F1 (weighted): {info['f1_weighted']:.4f}")
        print(
            f"CV ACC mean±std: {info['cv_mean_acc']:.4f} ± {info['cv_std_acc']:.4f}"
        )
        
if __name__ == "__main__":
    main()
