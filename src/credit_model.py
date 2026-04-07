import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 🔥 FORCE correct path
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Working directory:", BASE_DIR)
print("Saving to:", OUTPUT_DIR)

# =========================
# DATA
# =========================
np.random.seed(42)
n = 1000

credit_score = np.random.normal(650, 50, n)
loan_amount = np.random.normal(30000, 10000, n)

pd_vals = 1 / (1 + np.exp(-(0.01*(650-credit_score))))

df = pd.DataFrame({
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "PD": pd_vals
})

df["LGD"] = 0.4
df["EAD"] = df["loan_amount"]
df["ECL"] = df["PD"] * df["LGD"] * df["EAD"]

# =========================
# SAVE IMAGE
# =========================
plt.figure()
plt.hist(df["PD"], bins=30)
plt.title("PD Distribution")

img_path = os.path.join(OUTPUT_DIR, "credit_risk_dashboard.png")
plt.savefig(img_path)
plt.close()

print("Saved image:", img_path)

# =========================
# SAVE EXCEL
# =========================
excel_path = os.path.join(OUTPUT_DIR, "credit_portfolio_report.xlsx")
df.to_excel(excel_path, index=False)

print("Saved excel:", excel_path)
print("DONE")
