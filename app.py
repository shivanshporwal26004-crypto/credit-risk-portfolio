import streamlit as st
import subprocess
import os
from PIL import Image

st.title("📊 Credit Risk Dashboard")

if st.button("Run Model 🚀"):

    # Run model from correct folder
    subprocess.run(
        ["python3", "src/credit_model.py"],
        cwd=os.getcwd()
    )

    st.success("Model executed!")

    # Debug (important)
    if os.path.exists("outputs"):
        st.write("Files:", os.listdir("outputs"))
    else:
        st.error("Outputs folder not found")

    # Show image
    if os.path.exists("outputs/credit_risk_dashboard.png"):
        st.image("outputs/credit_risk_dashboard.png")
    else:
        st.error("Dashboard not found")

    # Download Excel
    if os.path.exists("outputs/credit_portfolio_report.xlsx"):
        with open("outputs/credit_portfolio_report.xlsx", "rb") as f:
            st.download_button("Download Excel", f)
