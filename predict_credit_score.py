# ===========================================================
# File: predict_credit_score.py
# Purpose: Predict credit scores for sample profiles (Excellent, Good, Fair, Bad)
# ===========================================================

import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# 1. Load Model and Scaler
# -------------------------------
MODEL_PATH = "ntc_credit_model_clean.pkl"
SCALER_PATH = "ntc_scaler.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    print("❌ Model or Scaler file not found! Please train or unwrap them first.")
    sys.exit(1)

print("🔍 Loading model and scaler...")
model: xgb.XGBClassifier = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded successfully!")

# -------------------------------
# 2. Define sample applicants
# -------------------------------
profiles = {
    "excellent": {
        "monthly_income": 150000,
        "age": 45,
        "job_stability_years": 12,
        "electricity_bill_ratio": 0.10,
        "gas_bill_ratio": 0.05,
        "mobile_bill_ratio": 0.10,
        "savings_ratio": 0.50,
        "avg_monthly_balance": 80000,
        "dependents": 1,
    },
    "good": {
        "monthly_income": 100000,
        "age": 35,
        "job_stability_years": 6,
        "electricity_bill_ratio": 0.15,
        "gas_bill_ratio": 0.08,
        "mobile_bill_ratio": 0.18,
        "savings_ratio": 0.30,
        "avg_monthly_balance": 40000,
        "dependents": 2,
    },
    "fair": {
        "monthly_income": 60000,
        "age": 30,
        "job_stability_years": 3,
        "electricity_bill_ratio": 0.22,
        "gas_bill_ratio": 0.10,
        "mobile_bill_ratio": 0.25,
        "savings_ratio": 0.15,
        "avg_monthly_balance": 15000,
        "dependents": 3,
    },
    "bad": {
        "monthly_income": 40000,
        "age": 25,
        "job_stability_years": 1,
        "electricity_bill_ratio": 0.28,
        "gas_bill_ratio": 0.14,
        "mobile_bill_ratio": 0.35,
        "savings_ratio": 0.05,
        "avg_monthly_balance": 5000,
        "dependents": 4,
    },
}

# Choose profile from CLI: e.g. python predict_credit_score.py good
if len(sys.argv) > 1:
    profile_key = sys.argv[1].lower()
    if profile_key not in profiles:
        print(f"⚠️ Unknown profile '{profile_key}'. Choose from: {list(profiles.keys())}")
        sys.exit(1)
    selected_profile = profiles[profile_key]
    print(f"📂 Running prediction for profile: {profile_key.upper()}")
else:
    selected_profile = profiles["excellent"]
    print("⚙️ No profile provided. Defaulting to: EXCELLENT")

new_data = pd.DataFrame([selected_profile])

print("\n🧾 Input data:")
print(new_data)

# -------------------------------
# 3. Align Columns to Training Features
# -------------------------------
try:
    expected_cols = scaler.feature_names_in_
except AttributeError:
    print("⚠️ Could not detect original feature names. Please specify manually.")
    sys.exit(1)

missing_cols = [c for c in expected_cols if c not in new_data.columns]
extra_cols = [c for c in new_data.columns if c not in expected_cols]

if missing_cols:
    print(f"⚠️ Missing columns in input: {missing_cols}")
    for col in missing_cols:
        new_data[col] = 0

if extra_cols:
    print(f"⚠️ Extra columns not used by model: {extra_cols}")
    new_data = new_data.drop(columns=extra_cols)

new_data = new_data[expected_cols]

# -------------------------------
# 4. Scale and Predict
# -------------------------------
X_scaled = scaler.transform(new_data)
risk_prob = model.predict_proba(X_scaled)[:, 1]

# Detect inverted model
flip = False
if risk_prob[0] > 0.5 and new_data.iloc[0]["monthly_income"] > 80000:
    flip = True
if flip:
    print("⚠️ Detected inverse probability mapping (higher income → higher risk). Fixing scale...")
    risk_prob = 1 - risk_prob

# Compute credit score
credit_score = 300 + (600 * (1 - risk_prob))

results = new_data.copy()
results["Risk_Prob"] = risk_prob.round(4)
results["Predicted_Credit_Score"] = credit_score.round(0)

print("\n📊 Predictions:")
print(results)

results.to_csv(f"predicted_credit_score_{list(profiles.keys())[list(profiles.values()).index(selected_profile)]}.csv", index=False)
print("\n💾 Results saved successfully.")

# -------------------------------
# 5. SHAP Explainability (Optional)
# -------------------------------
try:
    booster = model.get_booster()
    booster.feature_names = list(new_data.columns)
    explainer = shap.TreeExplainer(booster, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)

    print("\n🔍 SHAP feature contributions computed successfully!")
    shap.summary_plot(shap_values, new_data, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️ Skipping SHAP visualization: {e}")

# ===========================================================
# END OF FILE
# ===========================================================
