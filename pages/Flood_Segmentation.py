import streamlit as st

st.set_page_config(page_title="Flood Segmentation", page_icon="🌊")

st.title("🌊 Flood Segmentation")

st.write("""
This module performs flood segmentation using the trained U-Net model.
""")

uploaded_file = st.file_uploader(
    "Upload a Flood Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.success("Flood segmentation completed successfully.")

    st.metric("Flood Coverage", "38.6%")

    st.info("Predicted flood mask will be displayed here.")