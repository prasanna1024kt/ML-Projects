# ...existing code...
import argparse
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import joblib
import lightgbm as lgb

# import preprocess from train.py (train.py must guard execution with __name__ == "__main__")
from train import preprocess


def load_test_data(trans_path: Path, iden_path: Path | None = None) -> pd.DataFrame:
    
    if not trans_path.exists():
        raise FileNotFoundError(f"Test transaction file not found: {trans_path}")
    df_trans = pd.read_parquet(trans_path)
    if iden_path:
        if not iden_path.exists():
            raise FileNotFoundError(f"Test identity file not found: {iden_path}")
        df_iden = pd.read_parquet(iden_path)
        df = df_trans.merge(df_iden, on="TransactionID", how="left")
    else:
        df = df_trans
    return df


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Try joblib (pickled sklearn / sklearn-wrapped LGBM)
    try:
        model = joblib.load(model_path)
        print(f"Loaded model with joblib from {model_path}")
        return model
    except Exception:
        pass

    # Try native LightGBM booster (.bin/.model)
    try:
        booster = lgb.Booster(model_file=str(model_path))
        print(f"Loaded LightGBM Booster from {model_path}")
        return booster
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model file {model_path!s} with joblib and LightGBM: {e}"
        )

def predict_from_model(model, X: pd.DataFrame) -> pd.Series:
    """
    Robust prediction that fills any missing model features with -999 and
    returns a pd.Series of probabilities aligned with X.index.
    """
    X = X.copy()

    # sklearn-style estimator / pipeline with predict_proba
    if hasattr(model, "predict_proba"):

    # Get expected feature names
        if hasattr(model, "booster_"):
            feat_names = model.booster_.feature_name()
        else:
            feat_names = model.feature_name()

        # Align columns to training order
        X_reindexed = X.reindex(columns=feat_names, fill_value=-999)

        probs = model.predict_proba(X_reindexed)[:, 1]
        return pd.Series(probs, index=X.index)

    # sklearn wrapper around LightGBM (has booster_)
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        feat_names = model.booster_.feature_name()
        if not feat_names:
            raise RuntimeError("Model.booster_ has no feature names.")
        X_reindexed = X.reindex(columns=feat_names, fill_value=-999)
        X_arr = X_reindexed.astype("float32").values
        preds = model.booster_.predict(X_arr, raw_score=False)
        return pd.Series(preds, index=X.index)

    # native LightGBM Booster
    if isinstance(model, lgb.Booster):
        feat_names = model.feature_name()
        if not feat_names:
            raise RuntimeError("Loaded LightGBM Booster has no feature names.")
        X_reindexed = X.reindex(columns=feat_names, fill_value=-999)
        X_arr = X_reindexed.astype("float32").values
        preds = model.predict(X_arr, raw_score=False)
        return pd.Series(preds, index=X.index)

    # generic fallback
    preds = model.predict(X)
    return pd.Series(preds, index=X.index)


def main(
    model_path: Path,
    trans_path: Path,
    iden_path: Path | None,
    out_path: Path,
):
    df = load_test_data(trans_path, iden_path)
    ids = df["TransactionID"].copy() if "TransactionID" in df.columns else pd.Series(range(len(df)), name="TransactionID")

    df_proc = preprocess(df)

    drop_cols = [c for c in ("isFraud", "TransactionID") if c in df_proc.columns]
    X = df_proc.drop(columns=drop_cols)

    model = load_model(model_path)
    proba = predict_from_model(model, X)

    out_df = pd.DataFrame({"TransactionID": ids, "isFraud": proba})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
     # Predict using saved model on test dataset
    MODEL_FILE = Path('../model/lgb_model.bin')
    trans_path = Path('../datasets/test_transaction.parquet')
    iden_path = Path('../datasets/test_identity.parquet')
    out_path = Path('../model/predictions.csv')
    main(MODEL_FILE, trans_path, iden_path, out_path)
