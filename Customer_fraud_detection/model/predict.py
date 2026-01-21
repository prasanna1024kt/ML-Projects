from pathlib import Path
import logging
import traceback
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

# import preprocess from train.py (train.py must guard execution with __name__ == "__main__")
from .train import preprocess

app = Flask("Fraud_Detection_Service")
logging.basicConfig(level=logging.INFO)

# Resolve default model path relative to this file
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "lgb_model.bin"


def load_model(model_path: Path):
    model_path = Path(model_path)
    app.logger.info("Loading model from %s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    # try joblib first (sklearn pipeline / estimator)
    try:
        m = joblib.load(model_path)
        app.logger.info("Loaded model with joblib, type=%s", type(m))
        return m
    except Exception:
        app.logger.info("joblib.load failed, trying LightGBM Booster")

    # try native LightGBM booster
    try:
        booster = lgb.Booster(model_file=str(model_path))
        app.logger.info("Loaded LightGBM Booster, feature_count=%d", len(booster.feature_name() or []))
        return booster
    except Exception as e:
        app.logger.exception("Failed to load model: %s", e)
        raise RuntimeError(f"Failed to load model file {model_path!s}: {e}")


MODEL = None
try:
    MODEL = load_model(DEFAULT_MODEL_PATH)
except Exception:
    app.logger.exception("Model load failed at startup")


def _prepare_features_for_booster(X: pd.DataFrame, feat_names: list[str]):
    # fill missing expected features with sentinel -999 and order columns
    X_reindexed = X.reindex(columns=feat_names, fill_value=-999)
    return X_reindexed.astype("float32").values


def predict_from_model(model, X: pd.DataFrame):
    """
    Return numpy array of probabilities aligned with X.index.
    Handles:
      - sklearn-like estimators with predict_proba
      - sklearn wrapper around LightGBM (has booster_)
      - native lightgbm.Booster
      - generic fallback (predict)
    """
    X = X.copy()

    # sklearn-like pipeline/estimator
    if hasattr(model, "predict_proba"):

    # Align columns to model expectation
        if hasattr(model, "booster_"):
            feat_names = model.booster_.feature_name()
        else:
            feat_names = model.feature_name()

        if not feat_names:
            raise RuntimeError("Model has no stored feature names.")

        X_arr = _prepare_features_for_booster(X, feat_names)

        probs = model.predict_proba(pd.DataFrame(X_arr, columns=feat_names))[:, 1]
        return np.asarray(probs)

    # sklearn wrapper around LightGBM (e.g., LGBMClassifier)
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        feat_names = model.booster_.feature_name()
        if not feat_names:
            raise RuntimeError("Model.booster_ has no feature names.")
        X_arr = _prepare_features_for_booster(X, feat_names)
        preds = model.booster_.predict(X_arr, raw_score=False)
        return np.asarray(preds)

    # native LightGBM Booster
    if isinstance(model, lgb.Booster):
        feat_names = model.feature_name()
        if not feat_names:
            raise RuntimeError("Loaded LightGBM Booster has no feature names.")
        X_arr = _prepare_features_for_booster(X, feat_names)
        preds = model.predict(X_arr, raw_score=False)
        return np.asarray(preds)

    # fallback
    preds = model.predict(X)
    return np.asarray(preds)


@app.route("/health", methods=["GET"])
def health():
    ok = MODEL is not None
    return jsonify({"status": "ok" if ok else "error", "model_loaded": ok})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if MODEL is None:
            return jsonify({"error": "model not loaded"}), 500

        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error": "invalid json body"}), 400

        data = payload.get("data")
        if data is None:
            return jsonify({"error": "provide 'data' key with list[dict] or dict"}), 400

        # accept dict or list[dict]
        if isinstance(data, dict):
            df_in = pd.DataFrame([data])
        else:
            df_in = pd.DataFrame(data)

        # preprocess (must produce all training features where possible)
        df_proc = preprocess(df_in)

        # drop id/label if accidentally present before predict
        drop_cols = [c for c in ("isFraud", "TransactionID") if c in df_proc.columns]
        X = df_proc.drop(columns=drop_cols)

        # predict
        preds = predict_from_model(MODEL, X)
        preds_list = [float(x) for x in preds]

        # return TransactionID if present in input, otherwise index
        ids = df_in["TransactionID"].tolist() if "TransactionID" in df_in.columns else list(range(len(df_in)))

        return jsonify({"TransactionID": ids, "prediction": preds_list})

    except Exception as e:
        app.logger.exception("Prediction failed: %s", e)
        return jsonify({"error": str(e), "trace": traceback.format_exc().splitlines()[-5:]}), 500


if __name__ == "__main__":
    # run dev server
    app.run(host="0.0.0.0", port=9696, debug=True)