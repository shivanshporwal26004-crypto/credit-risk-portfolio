# Credit Risk Portfolio Monitor

## 📌 Overview

This project builds an end-to-end credit risk monitoring system using Python. It estimates Probability of Default (PD), Expected Credit Loss (ECL), and performs stress testing and portfolio analysis.

## ⚙️ Features

* PD Model using Logistic Regression
* ECL Calculation (PD × LGD × EAD)
* Stress Testing (macro scenarios)
* Migration Matrix
* Risk Dashboard

## 📊 Outputs

* Excel Report (credit_portfolio_report.xlsx)
* Dashboard Visualization (credit_risk_dashboard.png)

## 🛠️ Tech Stack

* Python (pandas, numpy, sklearn, matplotlib)
* Excel

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl scipy
python credit_portfolio_project.py
```

## 📈 Key Results

* Model AUC: ~0.85
* Gini: ~0.70
* Portfolio ECL estimated with stress scenarios

## 👨‍💻 Author

Shivansh Porwal
