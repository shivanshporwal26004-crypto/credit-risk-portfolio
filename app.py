import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.title("📊 Credit Risk Dashboard")

if st.button("Run Model 🚀"):

    # =========================
    # CREATE OUTPUT FOLDER
    # =========================
    os.makedirs("outputs", exist_ok=True)

    # =========================
    # GENERATE DATA
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

    img_path = "outputs/credit_risk_dashboard.png"
    plt.savefig(img_path)
    plt.close()

    # =========================
    # SAVE EXCEL
    # =========================
    excel_path = "outputs/credit_portfolio_report.xlsx"
    df.to_excel(excel_path, index=False)

    st.success("Model executed successfully!")

    # =========================
    # SHOW OUTPUTS
    # =========================
    if os.path.exists(img_path):
        st.subheader("📈 Dashboard")
        st.image(Image.open(img_path))
    else:
        st.error("Dashboard not generated")

    if os.path.exists(excel_path):
        with open(excel_path, "rb") as f:
            st.download_button(
                "📥 Download Excel Report",
                f,
                file_name="credit_report.xlsx"
            )
