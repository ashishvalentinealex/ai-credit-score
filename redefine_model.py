# repack_model.py
import joblib
import numpy as np

print("🔍 Loading old model...")
m = joblib.load("ntc_credit_model.pkl")

base_model = m["base_model"]
lr = m["calibration_lr"]

print("✅ Found base model and calibration layer!")

# Build a callable calibrated model wrapper
class CalibratedXGB:
    def __init__(self, base, lr):
        self.base = base
        self.lr = lr

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        p = self.lr.predict_proba(raw)[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# create wrapped model
model = CalibratedXGB(base_model, lr)

# save clean model
joblib.dump(model, "ntc_credit_model.pkl")
print("💾 Re-saved clean unified model successfully!")
