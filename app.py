import streamlit as st
import subprocess
import os
from PIL import Image

st.title("📊 Credit Risk Dashboard")

if st.button("Run Model 🚀"):

    # ✅ ALWAYS create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # ✅ Run model
    subprocess.run(["python3", "src/credit_model.py"])

    st.success("Model executed!")

    # ✅ Debug (see what's inside outputs)
    st.write("Files in outputs:", os.listdir("outputs"))

    # ✅ Show dashboard
    if os.path.exists("outputs/credit_risk_dashboard.png"):
        st.image("outputs/credit_risk_dashboard.png")
    else:
        st.error("Dashboard not generated")

    # ✅ Download Excel
    if os.path.exists("outputs/credit_portfolio_report.xlsx"):
        with open("outputs/credit_portfolio_report.xlsx", "rb") as f:
            st.download_button("Download Excel", f)
