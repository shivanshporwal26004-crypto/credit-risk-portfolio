import streamlit as st
import subprocess
import os
from PIL import Image

st.title("📊 Credit Risk Dashboard")

if st.button("Run Model 🚀"):

    # run model
    subprocess.run(["python3", "src/credit_model.py"])

    st.success("Model executed!")

    # check outputs
    if os.path.exists("outputs"):
        st.write("Files:", os.listdir("outputs"))

    # show image
    if os.path.exists("outputs/credit_risk_dashboard.png"):
        st.image("outputs/credit_risk_dashboard.png")
    else:
        st.error("Dashboard not found")

    # download excel
    if os.path.exists("outputs/credit_portfolio_report.xlsx"):
        with open("outputs/credit_portfolio_report.xlsx", "rb") as f:
            st.download_button("Download Excel", f)
