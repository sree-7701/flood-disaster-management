import streamlit as st

st.set_page_config(
    page_title="Flood Disaster Management System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Disaster Management System")

st.markdown("""
# Welcome

This AI-powered application provides:

- 🌊 Flood Segmentation using U-Net
- 📊 Damage Assessment
- 🚑 Resource Recommendation
- 📈 Model Performance Analysis

### 👈 Select a page from the sidebar.
""")

st.success("Application loaded successfully!")