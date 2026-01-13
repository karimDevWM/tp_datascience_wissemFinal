# Finally, replace the content of your main app.py with this clean version.

import streamlit as st
import sys
import os

# Local imports
from config import FILE_FINAL, FILE_CLUSTER
from data_loader import load_data
from views.audit import render_audit_section
from views.rfm import render_rfm_section
from views.clustering import render_clustering_section

# Page configuration
st.set_page_config(page_title="Project Dashboard", layout="wide")

st.title("Project Data Visualization")

# 1. Audit Section
render_audit_section()

# 2. Final Dataset & RFM Section
df_final = load_data(FILE_FINAL)
if df_final is not None:
    render_rfm_section(df_final)
else:
    st.error(f"File not found: {FILE_FINAL}. Please run the transformation pipeline first.")

# 3. Clustering Section
st.header("")  # Spacer
df_cluster = load_data(FILE_CLUSTER)
if df_cluster is not None:
    render_clustering_section(df_cluster)
else:
    st.error(f"File not found: {FILE_CLUSTER}. Please run the transformation pipeline first.")