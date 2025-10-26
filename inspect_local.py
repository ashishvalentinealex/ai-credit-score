# import joblib
# m = joblib.load("ntc_credit_model.pkl")
# print("Top-level type:", type(m))
# if isinstance(m, dict):
#     for k,v in m.items():
#         print(f"{k}: {type(v)}")
#         if isinstance(v, dict):
#             print("  Subkeys:", list(v.keys()))


import joblib
import xgboost as xgb
from pprint import pprint

class CalibratedXGB(xgb.XGBClassifier):
    pass

print("🔍 Loading model...")
model = joblib.load("ntc_credit_model.pkl")
print("Loaded type:", type(model))

print("\n📋 Listing all attributes in CalibratedXGB:")
attrs = {k: type(v).__name__ for k, v in vars(model).items()}
pprint(attrs)

# Try to locate any nested XGBClassifier inside attributes
for name, val in vars(model).items():
    if isinstance(val, xgb.XGBClassifier):
        print(f"\n✅ Found nested XGBClassifier inside attribute '{name}'")
        joblib.dump(val, "ntc_credit_model_clean.pkl")
        print("💾 Saved clean model → ntc_credit_model_clean.pkl")
        break
else:
    print("\n❌ No direct XGBClassifier found. Need to inspect printed attributes above.")
