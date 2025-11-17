# print_features.py
import pickle, json
dv_path = './model/heart_failure_model_decision_tree.bin'
with open(dv_path,'rb') as f:
    dv, model = pickle.load(f)
print(json.dumps(list(getattr(dv, "feature_names_", dv.get_feature_names_out())), indent=2))