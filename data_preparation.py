# =============================================
# File: data_preparation.py
# Purpose: Generate synthetic data for New-to-Credit (NTC) customers
# =============================================

import numpy as np
import pandas as pd

# ------------------------
# 1. Configuration
# ------------------------
np.random.seed(42)
N = 5000  # number of customers (was 2000)

# ------------------------
# 2. Generate synthetic customer features
# ------------------------
data = pd.DataFrame({
    # Demographics
    "age": np.random.randint(22, 45, size=N),
    "dependents": np.random.randint(0, 3, size=N),

    # Income & employment
    "monthly_income": np.random.normal(50000, 12000, size=N),
    "job_stability_years": np.random.uniform(0, 5, size=N),

    # Alternative data (utility & mobile payments)
    "electricity_bill_ratio": np.random.uniform(0.5, 1.0, size=N),
    "gas_bill_ratio": np.random.uniform(0.5, 1.0, size=N),
    "mobile_bill_ratio": np.random.uniform(0.6, 1.0, size=N),

    # Financial discipline indicators
    "savings_ratio": np.random.uniform(0.05, 0.5, size=N),
    "avg_monthly_balance": np.random.normal(20000, 5000, size=N),
})

# ------------------------
# 3. Create synthetic target label (0 = good, 1 = risky)
# ------------------------
data["target"] = (
    (data["monthly_income"] < 40000).astype(int) +
    (data["electricity_bill_ratio"] < 0.8).astype(int) +
    (data["gas_bill_ratio"] < 0.8).astype(int) +
    (data["mobile_bill_ratio"] < 0.85).astype(int) +
    (data["savings_ratio"] < 0.25).astype(int) +
    (data["job_stability_years"] < 2).astype(int)
)

# If ≥3 negative signals → risky customer
data["target"] = (data["target"] >= 3).astype(int)

# ------------------------
# 4. Quick sanity check
# ------------------------
print(" Data generation complete")
print(data.head(10))
print("\nGood vs Risky counts:")
print(data["target"].value_counts())

# ------------------------
# 5. Save dataset
# ------------------------
data.to_csv("ntc_credit_data.csv", index=False)
print(f"\nSaved ntc_credit_data.csv — {len(data)} rows created.")
