import os
import pickle
import logging
import streamlit as st
from typing import Optional
from sklearn.pipeline import Pipeline
from config import MODEL_FILES

def load_model(model_name: str) -> Optional[Pipeline]:
    """Load a trained model from disk."""
    try:
        if not os.path.exists(model_name):
            return None
        model = pickle.load(open(model_name, 'rb'))
        if isinstance(model, tuple):  # For backward compatibility
            return model[0]
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None
