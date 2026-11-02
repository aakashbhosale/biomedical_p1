"""
Comparative Evaluation of Data Normalization Techniques for ECG and EEG Signals
---------------------------------------------------------------------------
Runnable demonstration by default; accepts real CSV datasets for research runs.

CSV format:
- One row per observation/window/recording.
- Numeric feature columns (e.g., samples or extracted signal features).
- One target/label column.
- Optional separate ECG and EEG CSVs via command-line arguments.

Important:
The built-in demo data are synthetic and MUST NOT be reported as real biomedical findings.
For real data, keep subjects/records separated between train and test where applicable.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import boxcox
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def make_synthetic_dataset(kind: str, n_samples: int = 900, n_features: int = 32):
    """Generate a labelled toy dataset so the pipeline runs without external files."""
    rng = np.random.default_rng(RANDOM_STATE + (0 if kind == "ECG" else 1))
    y = np.repeat([0, 1, 2], n_samples // 3)
    if len(y) < n_samples:
        y = np.concatenate([y, rng.integers(0, 3, n_samples - len(y))])
    rng.shuffle(y)

    # Non-Gaussian positive baseline with class-dependent patterns.
    X = rng.exponential(scale=1.0, size=(n_samples, n_features))
    t = np.linspace(0, 2 * np.pi, n_features)
    for i, cls in enumerate(y):
        X[i] += (0.45 + 0.35 * cls) * (np.sin(t * (cls + 1)) + 1.2)
        if kind == "EEG":
            X[i] += 0.2 * np.cos(t * (cls + 2))
    # Add small measurement-like noise and a few missing entries.
    X += rng.normal(0, 0.08, X.shape)
    X = np.clip(X, 0.001, None)
    missing = rng.random(X.shape) < 0.002
    X[missing] = np.nan
    cols = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    return df


def load_dataset(csv_path: str | None, kind: str, label_column: str):
    if not csv_path:
        print(f"[INFO] {kind}: using SYNTHETIC demo data (not real biomedical data).")
        df = make_synthetic_dataset(kind)
    else:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"{kind} CSV not found: {path}")
        df = pd.read_csv(path)
        if label_column not in df.columns:
            raise ValueError(
                f"{kind} CSV must contain target column '{label_column}'. "
                f"Available columns: {list(df.columns)}"
            )

    y = df[label_column].copy()
    X = df.drop(columns=[label_column]).select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError(f"{kind}: no numeric feature columns found.")
    # Drop columns that are entirely missing; median imputation handles partial missingness.
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError(f"{kind}: all feature columns are empty/non-numeric.")
    # Remove rows with missing labels.
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if y.nunique() < 2:
        raise ValueError(f"{kind}: at least two target classes are required.")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))
    return X.to_numpy(dtype=float), y_encoded, encoder, list(X.columns)


def make_transformer(method: str, n_train: int):
    if method == "MinMax":
        return MinMaxScaler()
    if method == "Quantile":
        return QuantileTransformer(
            n_quantiles=max(1, min(500, n_train)),
            output_distribution="normal",
            random_state=RANDOM_STATE,
            subsample=100_000,
        )
    if method == "BoxCox":
        # PowerTransformer estimates lambda using training data only.
        # Box-Cox requires strictly positive inputs; shift is learned in pipeline per feature.
        return PositiveBoxCoxTransformer()
    raise ValueError(f"Unknown transform: {method}")


from sklearn.base import BaseEstimator, TransformerMixin


class PositiveBoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Feature-wise Box-Cox with training-derived positive shifts, fitted on training data only."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.imputer_ = SimpleImputer(strategy="median")
        Xi = self.imputer_.fit_transform(X)
        self.shifts_ = np.maximum(0.0, -np.nanmin(Xi, axis=0)) + 1e-6
        shifted = Xi + self.shifts_
        self.lambdas_ = []
        for j in range(shifted.shape[1]):
            col = shifted[:, j]
            if np.allclose(np.std(col), 0):
                self.lambdas_.append(1.0)
            else:
                _, lam = boxcox(col)
                self.lambdas_.append(lam)
        self.lambdas_ = np.asarray(self.lambdas_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xi = self.imputer_.transform(X)
        shifted = Xi + self.shifts_
        out = np.empty_like(shifted, dtype=float)
        for j, lam in enumerate(self.lambdas_):
            col = np.maximum(shifted[:, j], 1e-6)
            if np.allclose(np.std(col), 0):
                out[:, j] = col
            elif abs(lam) < 1e-8:
                out[:, j] = np.log(col)
            else:
                out[:, j] = (np.power(col, lam) - 1.0) / lam
        return out


def evaluate_dataset(kind, X, y, class_names, feature_names, test_size=0.25):
    # Stratification preserves class proportions; for real subject-based datasets,
    # replace this with a group/subject split to prevent leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    transforms = ["MinMax", "Quantile", "BoxCox"]
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVM-RBF": SVC(kernel="rbf", C=2.0, class_weight="balanced")
    }
    rows = []
    best_by_transform = {}

    for transform_name in transforms:
        for model_name, model in models.items():
            transformer = make_transformer(transform_name, len(X_train))
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("transform", transformer),
                ("classifier", clone(model)),
            ])
            # For BoxCox the transformer itself imputes; pipeline imputer is still useful
            # and ensures all other transforms receive complete data.
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            metrics = {
                "dataset": kind,
                "normalization": transform_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, pred),
                "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            }
            rows.append(metrics)
            key = (transform_name, model_name)
            best_by_transform[key] = (metrics["f1_macro"], pipeline, pred)

            print(
                f"{kind:3s} | {transform_name:8s} | {model_name:12s} | "
                f"Accuracy={metrics['accuracy']:.3f} | Macro-F1={metrics['f1_macro']:.3f}"
            )

    # Save confusion matrix and classification report for the highest macro-F1 configuration.
    best_key = max(best_by_transform, key=lambda k: best_by_transform[k][0])
    best_f1, best_pipeline, best_pred = best_by_transform[best_key]
    best_transform, best_model = best_key
    cm_path = RESULTS_DIR / f"{kind.lower()}_best_confusion_matrix.png"
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, display_labels=class_names, xticks_rotation="vertical",
        cmap=None, values_format="d"
    )
    disp.ax_.set_title(f"{kind}: Best demo/test split\n{best_transform} + {best_model}")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160, bbox_inches="tight")
    plt.close()

    report = classification_report(
        y_test, best_pred, target_names=class_names, zero_division=0
    )
    (RESULTS_DIR / f"{kind.lower()}_best_classification_report.txt").write_text(
        f"Best configuration: {best_transform} + {best_model}\n"
        f"Macro-F1: {best_f1:.4f}\n\n{report}",
        encoding="utf-8"
    )

    # Plot feature distribution snapshots for the first feature.
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    raw = X_train[:, 0]
    axes[0].hist(raw[np.isfinite(raw)], bins=30)
    axes[0].set_title("Original")
    for ax, method in zip(axes[1:], transforms):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("transform", make_transformer(method, len(X_train)))
        ])
        transformed = pipe.fit_transform(X_train)[:, 0]
        ax.hist(transformed[np.isfinite(transformed)], bins=30)
        ax.set_title(method)
    for ax in axes:
        ax.set_ylabel("Count")
    fig.suptitle(f"{kind}: Feature 1 distribution (training data)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{kind.lower()}_normalization_distributions.png", dpi=160)
    plt.close()

    return rows


def main():
    parser = argparse.ArgumentParser(description="ECG + EEG normalization comparison")
    parser.add_argument("--ecg", default=None, help="Path to ECG CSV; omit for synthetic demo")
    parser.add_argument("--eeg", default=None, help="Path to EEG CSV; omit for synthetic demo")
    parser.add_argument("--label-column", default="label", help="Target column name in both CSV files")
    args = parser.parse_args()

    all_rows = []
    for kind, path in [("ECG", args.ecg), ("EEG", args.eeg)]:
        X, y, encoder, feature_names = load_dataset(path, kind, args.label_column)
        class_names = [str(c) for c in encoder.classes_]
        print(f"\n--- {kind} dataset: {X.shape[0]} rows, {X.shape[1]} numeric features, "
              f"{len(class_names)} classes ---")
        all_rows.extend(evaluate_dataset(kind, X, y, class_names, feature_names))

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)

    # Plot macro-F1 for each model/transform, separately grouped by signal type.
    for kind in ["ECG", "EEG"]:
        subset = results[results["dataset"] == kind]
        pivot = subset.pivot(index="normalization", columns="model", values="f1_macro")
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title(f"{kind}: Macro-F1 comparison")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{kind.lower()}_metric_comparison.png", dpi=160)
        plt.close()

    print("\n=== RESULTS SUMMARY ===")
    print(results.sort_values(["dataset", "f1_macro"], ascending=[True, False]).round(4).to_string(index=False))
    print("\nSaved outputs in ./results/")
    print("NOTE: If you ran without --ecg and --eeg, these are synthetic demonstration results.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
