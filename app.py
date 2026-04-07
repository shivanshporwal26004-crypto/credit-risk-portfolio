import streamlit as st
import subprocess
import os
from PIL import Image

st.title("📊 Credit Risk Dashboard")

st.write("Click below to run full credit risk model")

if st.button("Run Model 🚀"):

    # Run your actual model
    subprocess.run(["python3", "src/credit_model.py"])

    st.success("Model executed successfully!")

    # Show dashboard image
    if os.path.exists("outputs/credit_risk_dashboard.png"):
        st.subheader("📈 Risk Dashboard")
        img = Image.open("outputs/credit_risk_dashboard.png")
        st.image(img)

    # Download Excel
    if os.path.exists("outputs/credit_portfolio_report.xlsx"):
        with open("outputs/credit_portfolio_report.xlsx", "rb") as f:
            st.download_button(
                "📥 Download Excel Report",
                f,
                file_name="credit_report.xlsx"
            )
