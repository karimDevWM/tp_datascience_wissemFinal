# This file is to manage your project paths.

import os

# Paths to data files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # One level up from dashboard/
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULT_DIR = os.path.join(CURRENT_DIR, 'result')
FILE_FINAL = os.path.join(RESULT_DIR, 'df_final.csv')
FILE_CLUSTER = os.path.join(RESULT_DIR, 'df_cluster.csv')