
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# Load Data
data = pd.read_csv("ntc_credit_data.csv")
print(" Data loaded:", data.shape)
print(data.head())

X = data.drop("target", axis=1)
y = data["target"]

# Add small random noise to labels (to avoid perfect separability)
flip_mask = np.random.rand(len(y)) < 0.05
y_noisy = y.copy()
y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
print(f" Introduced noise into labels for realism: {flip_mask.sum()} samples flipped")


# Split + Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train XGBoost Model
print("\n Training XGBoost model ...")
model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)
print(" XGBoost training complete!")

# Evaluate Model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("\nModel Performance:")
print("ROC-AUC Score:", round(auc, 3))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Convert Probability -> Credit Score (300–900)
credit_score = 900 - (600 * y_prob)
results = X_test.copy()
results["Risk_Prob"] = y_prob
results["Credit_Score"] = credit_score.round(0)

print("\nSample Predicted Credit Scores:")
print(results.head(10))

# SHAP Explainability (Stable for XGBoost ≥1.7)
print("\n Generating SHAP explanations (interventional mode)...")

# use booster to ensure compatibility
booster = model.get_booster()
booster.feature_names = list(X.columns)

# TreeExplainer with 'interventional' perturbation avoids coverage errors
explainer = shap.TreeExplainer(booster, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_scaled)

print(" SHAP values computed successfully!")

# --- SHAP Bar Summary Plot ---
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - NTC Credit Model")
plt.tight_layout()
plt.show()

# --- SHAP Full Summary Plot ---
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot (Feature Impact on Risk)")
plt.tight_layout()
plt.show()

# XGBoost Feature Importance
xgb.plot_importance(model, importance_type="gain")
plt.title("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.show()


# Individual Example Explanation
index = 5
print(f"\n Explaining Customer #{index}")
shap.initjs()
try:
    shap.force_plot(explainer.expected_value, shap_values[index, :], X_test.iloc[index, :])
except Exception:
    print(" Force plot works best in Jupyter/Notebook environments.")


