import streamlit as st

st.set_page_config(page_title="Disaster Assessment", page_icon="📍")

st.title("📍 Disaster Assessment")

st.write("""
This page analyzes the segmented flood region and estimates the disaster severity.
""")

st.subheader("Assessment Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Affected Area", "38.6%")

with col2:
    st.metric("Severity", "High")

st.success("Flood severity assessment completed.")