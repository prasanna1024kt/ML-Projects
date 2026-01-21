import lightgbm as lgb
import joblib, pickle, sys
from pathlib import Path
MODEL_FILE = Path("../model/lgb_model.bin")
p = Path("../model/lgb_model.bin")
#model = lgb.Booster(model_file=MODEL_FILE)
mld = joblib.load(p)
print(mld.feature_name_)