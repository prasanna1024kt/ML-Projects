from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd

# reuse preprocess from train.py (train.py must guard execution)
from train import preprocess

app = Flask("Fraud_Detection_Service")
MODEL_FILE = Path("../model/lgb_model.bin")


def load_model(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Try joblib (pickled sklearn / sklearn-wrapped LGBM / pipeline)
    try:
        m = joblib.load(path)
        print(f"Loaded model with joblib, type: {type(m)}")
        return m
    except Exception:
        pass

    # Try native LightGBM Booster (.bin/.model)
    try:
        booster = lgb.Booster(model_file=str(path))
        print(f"Loaded LightGBM Booster from {path}")
        return booster
    except Exception as e:
        raise RuntimeError(f"Failed to load model with joblib and LightGBM: {e}")


model = load_model(MODEL_FILE)

def load_model(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        m = joblib.load(path)
        print(f"Loaded model with joblib, type: {type(m)}")
        return m
    except Exception:
        pass

    try:
        booster = lgb.Booster(model_file=str(path))
        print(f"Loaded LightGBM Booster from {path}")
        print("Model feature count:", len(booster.feature_name()))
        print("First 20 model features:", booster.feature_name()[:20])
        return booster
    except Exception as e:
        raise RuntimeError(f"Failed to load model with joblib and LightGBM: {e}")

def prepare_input(data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    df_proc = preprocess(df)

    if isinstance(model, lgb.Booster):
        feat_names = model.feature_name()
        # diagnostics
        print("Preprocessed columns count:", len(df_proc.columns))
        print("Preprocessed columns (first 30):", list(df_proc.columns)[:30])
        missing = [f for f in feat_names if f not in df_proc.columns]
        if missing:
            print(f"Missing {len(missing)} features required by model. Showing first 50 missing:")
            print(missing[:50])
            # fail early with clear message
            raise ValueError(
                f"The input is missing {len(missing)} features required by the model. "
                "Ensure preprocess produces the same feature names as training."
            )
        # reindex (guarantees order) and pass numpy float32 to Booster
        df_proc = df_proc.reindex(columns=feat_names)
        X = df_proc.astype("float32").values
    else:
        drop_cols = [c for c in ("isFraud", "TransactionID") if c in df_proc.columns]
        X = df_proc.drop(columns=drop_cols)
    return X


def predict_fraud(data):
    X = prepare_input(data)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif isinstance(model, lgb.Booster):
        proba = model.predict(X)
    else:
        proba = model.predict(X)
    return proba


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        data = payload.get("data")
        if data is None:
            return jsonify({"error": "provide 'data' in JSON body"}), 400
        preds = predict_fraud(data)
        return jsonify({"prediction": pd.Series(preds).tolist()})
    except Exception as e:
        # log full traceback to console
        app.logger.exception("Prediction failed")
        # return minimal error to client
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run server: python test.py
    app.run(host="0.0.0.0", port=5000)