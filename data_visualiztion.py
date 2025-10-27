import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset

data = pd.read_csv("ntc_credit_data.csv")
print(" Data loaded successfully:", data.shape)
print(data.head())


# Basic summary
print("\n--- Summary Statistics ---")
print(data.describe())

print("\n--- Target distribution ---")
print(data['target'].value_counts())


# Distribution plots for key numerical features
num_cols = ["monthly_income", "savings_ratio", "avg_monthly_balance",
            "electricity_bill_ratio", "gas_bill_ratio", "mobile_bill_ratio"]

plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], bins=20, kde=True, color="skyblue")
    plt.title(f"{col}")
plt.tight_layout()
plt.show()


# Relationship of features vs target

plt.figure(figsize=(10, 6))
sns.countplot(x="target", data=data, palette=["#66BB6A", "#EF5350"])
plt.title("Distribution of Financially Good (0) vs Risky (1) Customers")
plt.xlabel("Target Category")
plt.ylabel("Count")
plt.show()

# Compare average feature values by target class
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x="target", y="monthly_income", palette=["#81C784", "#E57373"])
plt.title("Average Monthly Income by Risk Category")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="target", y="electricity_bill_ratio", palette="pastel")
plt.title("Electricity Bill Ratio by Risk Category")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
