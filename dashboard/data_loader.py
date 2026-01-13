# Create this file to handle system paths and data loading.

import streamlit as st
import pandas as pd
import sys
import os
from config import PROJECT_ROOT

# Add project root to sys.path so local `src` imports resolve
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Local package imports
from src.Extract import extract_dfs
from src.Explore import explore_dfs


@st.cache_data
def load_raw_data(data_dir):
    return extract_dfs(data_dir)


@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None
