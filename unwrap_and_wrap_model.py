# ===========================================================
# File: unwrap_and_save_model.py
# Purpose: fully unwrap CalibratedXGB → plain XGBClassifier
# ===========================================================

import joblib
import xgboost as xgb
import inspect

# placeholder so joblib can load
class CalibratedXGB(xgb.XGBClassifier):
    """dummy wrapper for loading legacy model"""
    pass

print("🔍 Loading model...")
model = joblib.load("ntc_credit_model.pkl")
print("Loaded type:", type(model))

# Try known attributes
for attr in ["base_estimator_", "model", "base_model", "_model", "wrapped_model"]:
    if hasattr(model, attr):
        inner = getattr(model, attr)
        print(f"✅ Found inner model at .{attr}: {type(inner)}")
        model = inner
        break

# Otherwise, try searching attributes dynamically
if not isinstance(model, xgb.XGBClassifier):
    for name, val in inspect.getmembers(model):
        if isinstance(val, xgb.XGBClassifier):
            print(f"✅ Found nested XGBClassifier at attribute '{name}'")
            model = val
            break

print("Final model type:", type(model))

# Verify it’s now an XGBClassifier
if not isinstance(model, xgb.XGBClassifier):
    raise TypeError("❌ Could not unwrap to an XGBClassifier. Inspect manually.")

# Save cleaned model
joblib.dump(model, "ntc_credit_model_clean.pkl")
print("💾 Saved clean model → ntc_credit_model_clean.pkl")
