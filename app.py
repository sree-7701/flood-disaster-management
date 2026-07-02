import streamlit as st

st.set_page_config(
    page_title="Flood Disaster Management System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Flood Disaster Management System")

st.markdown("""
## AI-Powered Flood Damage Assessment and Resource Recommendation

Welcome to the Flood Disaster Management System.

This application provides:

- 🌊 Flood Segmentation using U-Net
- 📍 Disaster Assessment
- 🚑 Intelligent Resource Recommendation
- 📊 Model Performance Analysis

Use the navigation menu on the left to explore the application.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("IoU", "0.4317")

with col2:
    st.metric("Dice Score", "0.6031")

with col3:
    st.metric("Accuracy", "87.98%")

st.divider()

st.info("👈 Select a page from the sidebar to begin.")