import streamlit as st
import os

st.title("Credit Risk App")

st.write("Click button to run model")

if st.button("Run Model"):
    os.system("python src/credit_model.py")
    st.success("Model executed!")

    if os.path.exists("outputs/credit_risk_dashboard.png"):
        st.image("outputs/credit_risk_dashboard.png")

    if os.path.exists("outputs/credit_portfolio_report.xlsx"):
        with open("outputs/credit_portfolio_report.xlsx", "rb") as f:
            st.download_button("Download Excel", f, "report.xlsx")
