# ...existing code...
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def preprocess(df):

    # Drop high missing columns
    missing = df.isnull().mean() * 100
    df = df.drop(columns=missing[missing > 85].index)

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns

    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    # Fill nulls
    df = df.fillna(-999)

    # Simple features
    if "TransactionAmt" in df.columns:
        df["amt_log"] = np.log1p(df["TransactionAmt"])

    if "card1" in df.columns and "TransactionID" in df.columns:
        df["card_txn_count"] = df.groupby("card1")["TransactionID"].transform("count")

    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600) % 24

    return df


def split_train_valid(df, frac: float = 0.8, sort_col: str = "TransactionDT"):
    """
    Return (train_df, valid_df). Sorts by sort_col if present, otherwise uses row order.
    """
    if sort_col in df.columns:
        df = df.sort_values(sort_col)
    n = len(df)
    split = int(n * frac)
    train_df = df.iloc[:split].copy()
    valid_df = df.iloc[split:].copy()
    return train_df, valid_df
def evaluate_model(y_true, y_proba, threshold: float = 0.5):
    """Print common binary classification metrics given true labels and predicted probabilities."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_proba)
    pr = average_precision_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"ROC-AUC: {roc:.5f}")
    print(f"PR-AUC : {pr:.5f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    return {"roc_auc": roc, "pr_auc": pr, "precision": prec, "recall": rec, "f1": f1, "confusion": cm}


def save_model(model, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df_trans = pd.read_parquet("../datasets/train_transaction.parquet")
    df_iden = pd.read_parquet("../datasets/train_identity.parquet")
    df_merged = df_trans.merge(df_iden, on="TransactionID", how="left")
    train_df = preprocess(df_merged)
    print("Train dataset loaded")
    print(f"Train dataset shape: {df_merged.shape}")
    print(f"processed dataset shape: {train_df.shape}")

    train, valid = split_train_valid(train_df, frac=0.8, sort_col="TransactionDT")

    X_train = train.drop(["isFraud", "TransactionID"], axis=1)
    y_train = train["isFraud"]
    X_valid = valid.drop(["isFraud", "TransactionID"], axis=1)
    y_valid = valid["isFraud"]

    '''
    model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    class_weight='balanced'
  )

    model.fit(X_train, y_train)
    print("Training completed")
    pred_valid = model.predict_proba(X_valid)[:,1]

    print("Validation ROC-AUC:", roc_auc_score(y_valid, pred_valid))
    print("Validation PR-AUC :", average_precision_score(y_valid, pred_valid))
    '''
    final_model = LGBMClassifier(
        num_leaves=63,
        max_depth=8,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        learning_rate=0.03,
        n_estimators=800,
        class_weight="balanced",
    )
    final_model.fit(X_train, y_train)
    final_pred_proba = final_model.predict_proba(X_valid)[:, 1]
    evaluate_model(y_valid, final_pred_proba, threshold=0.5)

    # save trained model
    save_model(final_model, "../model/lgb_model.bin")
    print("Final ROC-AUC:", roc_auc_score(y_valid, final_pred_proba))
    print("Final PR-AUC :", average_precision_score(y_valid, final_pred_proba))
