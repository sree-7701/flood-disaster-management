import streamlit as st

st.set_page_config(page_title="Model Performance", page_icon="📊")

st.title("📊 Model Performance")

st.write("""
Evaluation metrics of the trained U-Net model.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("IoU", "0.4317")

with col2:
    st.metric("Dice Score", "0.6031")

with col3:
    st.metric("Accuracy", "87.98%")

st.divider()

st.subheader("Performance Summary")

st.write("""
✔ Model: U-Net

✔ Dataset: DisasterM3

✔ Task: Flood Segmentation

✔ Evaluation completed successfully.
""")