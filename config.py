import logging
import streamlit as st

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit app configuration
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_FILES = {
    'diabetes': 'diabetes_model.sav',
    'heart': 'heart_disease_model.sav',
    'parkinsons': 'parkinsons_model.sav'
}
