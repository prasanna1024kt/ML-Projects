from flask import Flask, request, jsonify
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path

app = Flask(__name__)
MODEL_FILE = Path("../model/lgb_model.bin")
# Load LightGBM model saved as .bin
model = joblib.load(MODEL_FILE)

# Load expected feature order
model_columns = model.feature_name_


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    try:
        df = pd.DataFrame([data])

        # --- CRITICAL PART ---
        # Align incoming JSON to 361 model features
        print("Input data columns:", df.columns.tolist())
        for col in model_columns:
            if col not in df.columns:
                df[col] = -999      # default for missing fields

        df = df[model_columns]
        print("Aligned data columns:", df.columns.tolist())
        pred = model.predict(df)[0]

        return jsonify({
            "fraud_probability": float(pred),
            "is_fraud": int(pred > 0.3)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
