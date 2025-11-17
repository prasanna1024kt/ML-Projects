# ...existing code...
import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_FILE = './model/heart_failure_model_xgboost.bin'  
DATA_FILE = './Datasets/heart_failure_clinical_records.csv'  
THRESHOLD = 0.5 

app = Flask('predict')

# load model and dv
with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def _get_dv_feature_names(dv):
    if hasattr(dv, "get_feature_names_out"):
        return list(dv.get_feature_names_out())
    if hasattr(dv, "feature_names_"):
        return list(dv.feature_names_)
    if hasattr(dv, "get_feature_names"):
        return list(dv.get_feature_names())
    return []

def _preprocess_for_dv(df, dv):
    # mirror training preprocessing: fill numeric missing with median and create dummies
    df = df.fillna(df.median())
    df = pd.get_dummies(df, drop_first=True)
    feature_names = set(_get_dv_feature_names(dv))
    reduced = []
    for row in df.to_dict(orient='records'):
        reduced.append({k: v for k, v in row.items() if k in feature_names})
    return reduced

def compute_model_metrics(model, dv, data_path=DATA_FILE):
    if not os.path.exists(data_path):
        return {"accuracy": None, "roc_auc": None, "threshold": THRESHOLD}
    df = pd.read_csv(data_path)
    if "DEATH_EVENT" not in df.columns:
        return {"accuracy": None, "roc_auc": None, "threshold": THRESHOLD}

    X = df.drop(columns=["DEATH_EVENT"])
    y = df["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dicts = _preprocess_for_dv(X_test, dv)
    X_vect = dv.transform(dicts)

    score = None
    if hasattr(model, "predict_proba"):
        try:
            score = model.predict_proba(X_vect)[:, 1]
        except Exception:
            score = None
    if score is None and hasattr(model, "decision_function"):
        try:
            score = model.decision_function(X_vect)
        except Exception:
            score = None

    # accuracy using threshold if score available, otherwise model.predict
    if score is not None:
        y_pred = (score >= THRESHOLD).astype(int)
    else:
        y_pred = model.predict(X_vect)

    acc = accuracy_score(y_test, y_pred)

    roc = None
    if score is not None:
        try:
            roc = roc_auc_score(y_test, score)
        except Exception:
            roc = None

    return {"accuracy": float(acc), "roc_auc": (float(roc) if roc is not None else None), "threshold": THRESHOLD}

# compute metrics once at startup
MODEL_METRICS = compute_model_metrics(model, dv)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """
    Predict heart failure risk for one or more records.
    Response includes for each record:
      - prediction: probability (float) if available else integer class
      - heart_failure: boolean (prediction >= threshold)
      - threshold: threshold used
    Also returns model_metrics.
    """
    # Accept GET for quick test (single record via query params) or JSON body
    if request.method == 'GET':
        customer = dict(request.args)
        for k, v in list(customer.items()):
            try:
                customer[k] = float(v) if ('.' in v or 'e' in v.lower()) else int(v)
            except Exception:
                pass
        records = [customer]
    else:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "expected JSON body with 'data' key or raw record"}), 400
        data = payload.get("data", payload)
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return jsonify({"error": "'data' must be an object or list"}), 400

    dicts = _preprocess_for_dv(pd.DataFrame(records), dv)
    X_vect = dv.transform(dicts)

    # get preds and scores
    preds_class = model.predict(X_vect).tolist()
    scores = None
    if hasattr(model, "predict_proba"):
        try:
            scores = model.predict_proba(X_vect)[:, 1].tolist()
        except Exception:
            scores = None
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_vect).tolist()
        except Exception:
            scores = None

    # build per-record results including heart_failure flag and threshold
    results = []
    n = len(preds_class)
    for i in range(n):
        prob = scores[i] if scores is not None else None
        if prob is not None:
            hf = bool(prob >= THRESHOLD)
            pred_value = float(prob)
        else:
            hf = bool(preds_class[i] == 1)
            pred_value = int(preds_class[i])
        results.append({
            "prediction": pred_value,
            "heart_failure": hf,
            "threshold": THRESHOLD,
            "class_label": int(preds_class[i])
        })

    response = {
        "results": results,
        "model_metrics": MODEL_METRICS
    }
    if scores is not None:
        response["scores"] = scores

    return jsonify(response), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Return stored model metrics (accuracy, roc_auc, threshold)."""
    return jsonify(MODEL_METRICS), 200

@app.route("/features", methods=["GET"])
def features():
    """Return the feature names the DictVectorizer expects."""
    return jsonify({"features": _get_dv_feature_names(dv)}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)