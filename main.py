import os
import time
import logging
import streamlit as st
from streamlit_option_menu import option_menu
from config import MODEL_FILES
from model_utils import load_model
from model_training import train_diabetes_model, train_heart_model, train_parkinsons_model
from ui_components import diabetes_page, heart_page, parkinsons_page

def main():
    """Main application function."""
    diabetes_model = None
    heart_model = None
    parkinsons_model = None

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center">
            <h1 style="color:#4a8cff; margin-bottom:0">HealthGuard AI</h1>
            <p style="color:#666; margin-top:0">Comprehensive Health Assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        selected = option_menu(
            menu_title=None,
            options=['Diabetes', 'Heart Disease', "Parkinson's"],
            icons=['activity', 'heart-pulse', 'brain'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"font-size": "16px"}, 
                "nav-link": {"font-size": "16px", "margin": "5px", "--hover-color": "#f0f2f6"},
                "nav-link-selected": {"background-color": "#4a8cff"},
            }
        )
        
        st.markdown("---")
        
        with st.expander("ℹ️ About This App"):
            st.write("""
            **HealthGuard AI** provides preliminary risk assessments for:
            - Diabetes Mellitus
            - Coronary Heart Disease
            - Parkinson's Disease
            
            **Methodology:**
            - Uses machine learning models trained on clinical datasets
            - Provides risk estimates with confidence levels
            - Offers personalized health recommendations
            
            **Disclaimer:**
            This tool is for informational purposes only and not a substitute for professional medical advice, 
            diagnosis, or treatment. Always seek the advice of your physician or other qualified health 
            provider with any questions you may have regarding a medical condition.
            """)
        
        st.markdown("### Model Status")
        diabetes_status = "✅ Loaded" if os.path.exists(MODEL_FILES['diabetes']) else "❌ Not Found"
        heart_status = "✅ Loaded" if os.path.exists(MODEL_FILES['heart']) else "❌ Not Found"
        parkinsons_status = "✅ Loaded" if os.path.exists(MODEL_FILES['parkinsons']) else "❌ Not Found"
        
        st.markdown(f"""
        - Diabetes Model: {diabetes_status}
        - Heart Disease Model: {heart_status}
        - Parkinson's Model: {parkinsons_status}
        """)
        
        st.markdown("""
        <div style="text-align:center; margin-top:50px; color:#888; font-size:12px">
            <p>Developed by HealthGuard AI Team</p>
            <p>Version 1.1.0</p>
        </div>
        """, unsafe_allow_html=True)

    diabetes_model = load_model(MODEL_FILES['diabetes'])
    heart_model = load_model(MODEL_FILES['heart'])
    parkinsons_model = load_model(MODEL_FILES['parkinsons'])

    if diabetes_model is None:
        with st.spinner("Training diabetes prediction model..."):
            diabetes_model = train_diabetes_model()
    
    if heart_model is None:
        with st.spinner("Training heart disease prediction model..."):
            heart_model = train_heart_model()
    
    if parkinsons_model is None:
        with st.spinner("Training Parkinson's disease prediction model..."):
            parkinsons_model = train_parkinsons_model()

    if selected == 'Diabetes':
        if diabetes_model is not None:
            diabetes_page(diabetes_model)
        else:
            st.error("Diabetes prediction model failed to load. Please try again later.")
            logging.error("Diabetes model not available for prediction")

    elif selected == 'Heart Disease':
        if heart_model is not None:
            heart_page(heart_model)
        else:
            st.error("Heart disease prediction model failed to load. Please try again later.")
            logging.error("Heart model not available for prediction")

    elif selected == "Parkinson's":
        if parkinsons_model is not None:
            parkinsons_page(parkinsons_model)
        else:
            st.error("Parkinson's disease prediction model failed to load. Please try again later.")
            logging.error("Parkinson's model not available for prediction")

    st.markdown("---")
    with st.expander("💬 Provide Feedback"):
        feedback = st.text_area("Help us improve HealthGuard AI by sharing your feedback")
        if st.button("Submit Feedback"):
            if feedback.strip():
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open("feedback.txt", "a") as f:
                    f.write(f"{timestamp} - {selected} - {feedback}\n")
                st.success("Thank you for your feedback!")
                logging.info(f"Feedback received for {selected}: {feedback[:50]}...")
            else:
                st.warning("Please enter your feedback before submitting")

if __name__ == "__main__":
    main()
