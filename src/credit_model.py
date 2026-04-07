# =============================================================
# INTEGRATED CREDIT PORTFOLIO MANAGEMENT SYSTEM (FIXED VERSION)
# =============================================================

import os
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ================================
# STEP 1: DATA GENERATION
# ================================
n = 2000

age = np.random.randint(22, 65, n)
income = np.random.normal(55000, 20000, n).clip(15000, 200000)
loan_amount = np.random.normal(40000, 15000, n).clip(5000, 150000)
credit_score = np.random.normal(680, 80, n).clip(300, 850)
debt_to_income = loan_amount / income
employment_yrs = np.random.exponential(5, n)
num_past_dues = np.random.poisson(0.4, n)

# Default simulation
log_odds = (-0.008 * credit_score + 2.5 * debt_to_income + 0.4 * num_past_dues)
true_pd = 1 / (1 + np.exp(-log_odds))
default = (np.random.rand(n) < true_pd).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'debt_to_income': debt_to_income,
    'employment_yrs': employment_yrs,
    'num_past_dues': num_past_dues,
    'default': default
})

# ================================
# STEP 2: LOGISTIC REGRESSION
# ================================
features = ['credit_score', 'debt_to_income', 'employment_yrs', 'num_past_dues']
X = df[features]
y = df['default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

df['PD'] = model.predict_proba(X_scaled)[:, 1]

# ================================
# STEP 3: ECL CALCULATION
# ================================
df['LGD'] = 0.4
df['EAD'] = df['loan_amount']
df['ECL'] = df['PD'] * df['LGD'] * df['EAD']

# ================================
# STEP 4: SIMPLE DASHBOARD
# ================================
plt.figure(figsize=(10, 6))
plt.hist(df['PD'], bins=30)
plt.title("Probability of Default Distribution")
plt.xlabel("PD")
plt.ylabel("Count")

plt.savefig("outputs/credit_risk_dashboard.png")
plt.close()

# ================================
# STEP 5: EXPORT EXCEL
# ================================
df.to_excel("outputs/credit_portfolio_report.xlsx", index=False)

print("DONE: Files saved in outputs/")
