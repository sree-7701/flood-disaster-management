import streamlit as st

st.set_page_config(page_title="Resource Recommendation", page_icon="🚑")

st.title("🚑 Resource Recommendation")

st.write("""
Recommended emergency resources based on flood severity.
""")

priority = st.selectbox(
    "Flood Severity",
    ["Low", "Medium", "High"]
)

if st.button("Generate Recommendation"):

    if priority == "Low":
        st.success("""
        • 1 Rescue Team

        • 1 Medical Team

        • Food Packets

        • Drinking Water
        """)

    elif priority == "Medium":
        st.warning("""
        • 3 Rescue Teams

        • 2 Boats

        • 2 Medical Teams

        • Relief Camp
        """)

    else:
        st.error("""
        • 5 Rescue Teams

        • 4 Rescue Boats

        • 3 Medical Teams

        • Temporary Shelter

        • Food & Water Supply

        • Emergency Vehicles
        """)