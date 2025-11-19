# ...existing code...
import os
import pandas as pd
import pickle
import sys
import math
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
def preprocess_data(data):  
    # Handle missing values
    data = data.fillna(data.median())
    
    # Convert categorical variables to dummy/indicator variables
    data = pd.get_dummies(data, drop_first=True)
    
    return data
def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='random_forest'):  
    """
    Train a model using DictVectorizer to convert DataFrame -> feature matrix.
    Returns trained model and fitted DictVectorizer.
    """
    dv = DictVectorizer(sparse=False)
    X_train_dicts = X_train.to_dict(orient='records')
    X_train_vect = dv.fit_transform(X_train_dicts)

    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier( random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    # FIX: fit on the vectorized features
    model.fit(X_train_vect, y_train)
    return model,dv

def evaluate_model(model, dv, X_test, y_test):
    X_test_dicts = X_test.to_dict(orient='records')
    X_test_vect = dv.transform(X_test_dicts)

    # FIX: predict on vectorized test set
    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)

    # handle models without predict_proba
    prob_score = None
    if hasattr(model, "predict_proba"):
        try:
            prob_score = model.predict_proba(X_test_vect)[:, 1]
        except Exception:
            prob_score = None
    if prob_score is None and hasattr(model, "decision_function"):
        try:
            prob_score = model.decision_function(X_test_vect)
        except Exception:
            prob_score = None

    if prob_score is not None:
        try:
            roc_auc = roc_auc_score(y_test, prob_score)
        except Exception:
            roc_auc = math.nan
    else:
        roc_auc = math.nan

    return float(accuracy), float(roc_auc)

def save_model(output_file,model, dv,):
    # ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file,'wb') as f_out: 
        pickle.dump((dv,model),f_out)
    
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    model_type = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    data = load_data('./Datasets/heart_failure_clinical_records.csv')
    data = preprocess_data(data)
    output_file = './model/heart_failure_model.bin'
    # Split data
    X_train, X_test, y_train, y_test = split_data(data, target_column='DEATH_EVENT')
    available_models = ['logistic_regression','random_forest','decision_tree','xgboost']

    if model_type == 'all':
        results = {}
        for m in available_models:
            model, dv = train_model(X_train, y_train, model_type=m)
            acc, roc = evaluate_model(model, dv, X_test, y_test)
            results[m] = {"accuracy": acc, "roc_auc": (roc if not math.isnan(roc) else None)}
            # save each model artifact
            out_path = f'./model/heart_failure_model_{m}.bin'
            save_model(out_path, model, dv)
        # print JSON to stdout and exit (avoid printing single-model vars which don't exist)
        print(json.dumps(results, indent=2))
        sys.exit(0)
    else:
        if model_type not in available_models:
            raise SystemExit(f"Unknown model_type '{model_type}'. Choose from {available_models} or 'all'.")

        model, dv = train_model(X_train, y_train, model_type=model_type)
        accuracy, roc_auc = evaluate_model(model, dv, X_test, y_test)

        # save trained model + dv
        output_file = f'./model/heart_failure_model_{model_type}.bin'
        save_model(output_file, model, dv)

        print(f"Saved model -> {output_file}")
        print(f"Model Accuracy: {accuracy:.4f}")
        print(f"Model ROC-AUC: {roc_auc:.4f}")
