import streamlit as st

# Define the pages
main_page = st.Page("pages/CC_parcellation.py", title="CC Parcellation")
run_page = st.Page("pages/parc_setup.py", title="Parcellation Setup")
tmg_page = st.Page("pages/TMG_info.py", title="TMG Info")

# Set up navigation
pg = st.navigation([main_page, run_page, tmg_page])

with st.sidebar.expander(":small[Interface tips]"):
    st.markdown("""
    - :small[Refresh the page to reset the session]
    - :small[During processing, use the top-right activity indicator to monitor or stop execution]
    """)

# Run the selected page
pg.run()