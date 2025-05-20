import streamlit as st
import numpy as np
import pandas as pd
import time
import logging
from sklearn.pipeline import Pipeline

def show_prediction_result(prediction: np.ndarray, 
                          proba: np.ndarray, 
                          positive_result: str, 
                          negative_result: str) -> None:
    """Display prediction results with confidence and recommendations."""
    with st.spinner('Analyzing health data...'):
        time.sleep(1.5)
    
    confidence = max(proba[0]) * 100
    if prediction[0] == 1:
        st.error(positive_result, icon="⚠️")
        st.metric("Confidence Level", f"{confidence:.1f}%")
        with st.expander("📌 Recommended Actions"):
            st.markdown("""
            - Consult a healthcare professional immediately
            - Schedule follow-up diagnostic tests
            - Monitor symptoms closely
            - Consider lifestyle changes (diet, exercise)
            - Review family medical history
            - Track relevant health metrics regularly
            """)
    else:
        st.success(negative_result, icon="✅")
        st.metric("Confidence Level", f"{confidence:.1f}%")
        with st.expander("💡 Health Maintenance Tips"):
            st.markdown("""
            - Continue regular health check-ups
            - Maintain balanced diet and exercise routine
            - Monitor risk factors periodically
            - Stay within healthy weight range
            - Manage stress levels
            - Avoid smoking and limit alcohol consumption
            """)

def validate_diabetes_inputs(inputs: dict) -> bool:
    """Validate diabetes assessment inputs."""
    if inputs['Glucose'] < 40 or inputs['Glucose'] > 400:
        st.warning("Glucose level should be between 40-400 mg/dL")
        return False
    if inputs['BloodPressure'] < 20 or inputs['BloodPressure'] > 180:
        st.warning("Blood pressure should be between 20-180 mmHg")
        return False
    if inputs['BMI'] < 10 or inputs['BMI'] > 60:
        st.warning("BMI should be between 10-60")
        return False
    return True

def diabetes_page(model: Pipeline) -> None:
    """Diabetes risk assessment page."""
    st.title('🔍 Diabetes Risk Assessment')
    
    with st.expander("ℹ️ About This Assessment", expanded=False):
        st.write("""
        This tool evaluates your risk for diabetes based on key health indicators.
        Diabetes is a chronic condition that affects how your body processes blood sugar.
        Early detection can help prevent complications.
        
        **Clinical Ranges:**
        - Normal Glucose: 70-99 mg/dL (fasting)
        - Prediabetes: 100-125 mg/dL
        - Diabetes: ≥126 mg/dL
        """)
    
    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Basic Information")
            pregnancies = st.slider('Number of Pregnancies', 0, 20, 1, 
                                   help="Number of times pregnant (if applicable)")
            age = st.slider('Age (years)', 1, 120, 30)
            
        with col2:
            st.subheader("Blood Metrics")
            glucose = st.slider('Glucose Level (mg/dL)', 0, 400, 100,
                              help="Plasma glucose concentration from 2 hours in an oral glucose tolerance test")
            bp = st.slider('Blood Pressure (mmHg)', 0, 180, 70,
                         help="Diastolic blood pressure (mm Hg)")
            insulin = st.slider('Insulin Level (μU/mL)', 0, 900, 80,
                              help="2-Hour serum insulin (mu U/ml)")
            
        with col3:
            st.subheader("Body Composition")
            skin_thickness = st.slider('Skin Thickness (mm)', 0, 100, 20,
                                     help="Triceps skin fold thickness (mm)")
            bmi = st.slider('Body Mass Index (BMI)', 0.0, 70.0, 25.0, 0.1,
                          help="Weight in kg/(height in m)^2")
            dpf = st.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.5, 0.01,
                          help="Diabetes pedigree function (genetic influence)")

        submitted = st.form_submit_button('Assess Diabetes Risk', type="primary", use_container_width=True)
        
        if submitted:
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': bp,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            if not validate_diabetes_inputs(input_data):
                return
                
            try:
                input_array = np.asarray(list(input_data.values())).reshape(1, -1)
                prediction = model.predict(input_array)
                proba = model.predict_proba(input_array)
                
                show_prediction_result(
                    prediction, proba,
                    "High risk of diabetes detected. Please consult your doctor for further evaluation.",
                    "Low risk of diabetes detected. Maintain healthy lifestyle habits!"
                )
                
                if hasattr(model.named_steps['classifier'], 'coef_'):
                    st.subheader("Key Contributing Factors")
                    coef = model.named_steps['classifier'].coef_[0]
                    features = list(input_data.keys())
                    importance = pd.DataFrame({'Feature': features, 'Importance': coef})
                    importance = importance.sort_values('Importance', ascending=False)
                    st.bar_chart(importance.set_index('Feature'))
                
            except Exception as e:
                st.error(f'Error in prediction: {str(e)}')
                logging.error(f"Diabetes prediction error: {str(e)}")

def validate_heart_inputs(inputs: dict) -> bool:
    """Validate heart disease assessment inputs."""
    if inputs['trestbps'] < 50 or inputs['trestbps'] > 250:
        st.warning("Resting blood pressure should be between 50-250 mmHg")
        return False
    if inputs['chol'] < 100 or inputs['chol'] > 600:
        st.warning("Cholesterol level should be between 100-600 mg/dL")
        return False
    if inputs['thalach'] < 50 or inputs['thalach'] > 250:
        st.warning("Maximum heart rate should be between 50-250 bpm")
        return False
    return True

def heart_page(model: Pipeline) -> None:
    """Heart disease risk assessment page."""
    st.title('❤️ Heart Disease Risk Assessment')
    
    with st.expander("ℹ️ About This Assessment", expanded=False):
        st.write("""
        This tool evaluates your cardiovascular health risk based on clinical parameters.
        Heart disease is the leading cause of death worldwide.
        Early detection can significantly improve outcomes.
        
        **Clinical Ranges:**
        - Normal BP: <120/80 mmHg
        - Elevated BP: 120-129/<80 mmHg
        - High BP: ≥130/80 mmHg
        """)
    
    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Demographics")
            age = st.slider('Age (years)', 1, 120, 50)
            sex = st.radio('Sex', ['Male', 'Female'], horizontal=True)
            
        with col2:
            st.subheader("Vital Signs")
            trestbps = st.slider('Resting Blood Pressure (mmHg)', 50, 250, 120,
                               help="Resting blood pressure (in mm Hg on admission to the hospital)")
            thalach = st.slider('Maximum Heart Rate Achieved', 50, 250, 150,
                              help="Maximum heart rate achieved during exercise")
            oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 10.0, 1.0, 0.1,
                              help="ST depression induced by exercise relative to rest")
            
        with col3:
            st.subheader("Blood Work")
            chol = st.slider('Serum Cholesterol (mg/dL)', 100, 600, 200,
                           help="Serum cholesterol in mg/dl")
            fbs = st.radio('Fasting Blood Sugar > 120 mg/dL', ['No', 'Yes'], horizontal=True,
                         help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Symptoms")
            cp = st.select_slider('Chest Pain Type', 
                                 options=['Typical Angina', 'Atypical Angina', 
                                         'Non-Anginal Pain', 'Asymptomatic'],
                                 value='Atypical Angina',
                                 help="""1: typical angina, 
                                       2: atypical angina, 
                                       3: non-anginal pain, 
                                       4: asymptomatic""")
            exang = st.radio('Exercise Induced Angina', ['No', 'Yes'], horizontal=True,
                           help="Exercise induced angina (1 = yes; 0 = no)")
            
        with col2:
            st.subheader("ECG Results")
            restecg = st.select_slider('Resting Electrocardiographic Results', 
                                      options=['Normal', 'ST-T Abnormality', 
                                              'Left Ventricular Hypertrophy'],
                                      value='Normal',
                                      help="""0: normal,
                                            1: having ST-T wave abnormality,
                                            2: showing probable or definite left ventricular hypertrophy""")
            slope = st.select_slider('Slope of Peak Exercise ST Segment', 
                                   options=['Upsloping', 'Flat', 'Downsloping'],
                                   value='Flat',
                                   help="""1: upsloping,
                                         2: flat,
                                         3: downsloping""")
            
        with col3:
            st.subheader("Advanced Metrics")
            ca = st.select_slider('Number of Major Vessels Colored by Fluoroscopy', 
                                options=['0', '1', '2', '3', '4'],
                                value='0',
                                help="Number of major vessels (0-4) colored by fluoroscopy")
            thal = st.select_slider('Thalassemia', 
                                  options=['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'],
                                  value='Normal',
                                  help="""3 = normal;
                                        6 = fixed defect;
                                        7 = reversible defect;
                                        other values are considered unknown""")

        submitted = st.form_submit_button('Assess Heart Disease Risk', type="primary", use_container_width=True)
        
        if submitted:
            cp_mapping = {'Typical Angina': 1, 'Atypical Angina': 2, 
                         'Non-Anginal Pain': 3, 'Asymptomatic': 4}
            restecg_mapping = {'Normal': 0, 'ST-T Abnormality': 1, 
                              'Left Ventricular Hypertrophy': 2}
            slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
            thal_mapping = {'Normal': 3, 'Fixed Defect': 6, 
                           'Reversible Defect': 7, 'Unknown': 2}
            
            input_data = {
                'age': age,
                'sex': 1 if sex == 'Male' else 0,
                'cp': cp_mapping[cp],
                'trestbps': trestbps,
                'chol': chol,
                'fbs': 1 if fbs == 'Yes' else 0,
                'restecg': restecg_mapping[restecg],
                'thalach': thalach,
                'exang': 1 if exang == 'Yes' else 0,
                'oldpeak': oldpeak,
                'slope': slope_mapping[slope],
                'ca': int(ca),
                'thal': thal_mapping[thal]
            }
            
            if not validate_heart_inputs(input_data):
                return
                
            try:
                input_array = np.asarray(list(input_data.values())).reshape(1, -1)
                prediction = model.predict(input_array)
                proba = model.predict_proba(input_array)
                
                show_prediction_result(
                    prediction, proba,
                    "Elevated heart disease risk detected. Please seek medical advice for comprehensive evaluation.",
                    "Your heart disease risk appears low. Continue healthy cardiovascular practices!"
                )
                
                if hasattr(model.named_steps['classifier'], 'coef_'):
                    st.subheader("Key Contributing Factors")
                    coef = model.named_steps['classifier'].coef_[0]
                    features = list(input_data.keys())
                    importance = pd.DataFrame({'Feature': features, 'Importance': coef})
                    importance = importance.sort_values('Importance', ascending=False)
                    st.bar_chart(importance.set_index('Feature'))
                
            except Exception as e:
                st.error(f'Error in prediction: {str(e)}')
                logging.error(f"Heart prediction error: {str(e)}")

def validate_parkinsons_inputs(inputs: dict) -> bool:
    """Validate Parkinson's disease assessment inputs."""
    if inputs['MDVP:Fo(Hz)'] < 80 or inputs['MDVP:Fo(Hz)'] > 300:
        st.warning("Average vocal fundamental frequency should be between 80-300 Hz")
        return False
    if inputs['MDVP:Fhi(Hz)'] < 100 or inputs['MDVP:Fhi(Hz)'] > 600:
        st.warning("Maximum vocal fundamental frequency should be between 100-600 Hz")
        return False
    if inputs['HNR'] < 0 or inputs['HNR'] > 40:
        st.warning("Harmonics-to-noise ratio should be between 0-40 dB")
        return False
    return True

def parkinsons_page(model: Pipeline) -> None:
    """Parkinson's disease risk assessment page."""
    st.title('🧠 Parkinson\'s Disease Risk Assessment')
    
    with st.expander("ℹ️ About This Assessment", expanded=False):
        st.write("""
        This analysis evaluates vocal measurements to assess Parkinson's disease risk.
        Parkinson's is a progressive nervous system disorder that affects movement.
        Early detection can help manage symptoms more effectively.
        
        **Note:** This assessment is based on voice analysis and should be
        combined with clinical evaluation for accurate diagnosis.
        """)
    
    with st.form("parkinsons_form"):
        st.subheader("Vocal Fundamental Frequency")
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.slider('Average Vocal Fundamental Frequency (MDVP:Fo(Hz))', 
                          min_value=80.0, max_value=300.0, value=150.0, step=1.0,
                          help="Average vocal fundamental frequency in Hz")
        with col2:
            fhi = st.slider('Maximum Vocal Fundamental Frequency (MDVP:Fhi(Hz))', 
                           min_value=100.0, max_value=600.0, value=180.0, step=1.0,
                           help="Maximum vocal fundamental frequency in Hz")
        with col3:
            flo = st.slider('Minimum Vocal Fundamental Frequency (MDVP:Flo(Hz))', 
                           min_value=50.0, max_value=300.0, value=100.0, step=1.0,
                           help="Minimum vocal fundamental frequency in Hz")
        
        st.subheader("Voice Instability Measures")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            jitter_percent = st.slider('Jitter Percentage (MDVP:Jitter(%))', 
                                     min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f",
                                     help="Variation in fundamental frequency")
        with col2:
            jitter_abs = st.slider('Absolute Jitter (MDVP:Jitter(Abs))', 
                                 min_value=0.00001, max_value=0.001, value=0.00003, step=0.00001, format="%.5f",
                                 help="Absolute jitter measurement")
        with col3:
            rap = st.slider('Relative Amplitude Perturbation (MDVP:RAP))', 
                           min_value=0.001, max_value=0.1, value=0.003, step=0.001, format="%.3f",
                           help="Relative amplitude perturbation")
        with col4:
            ppq = st.slider('Period Perturbation Quotient (MDVP:PPQ))', 
                           min_value=0.001, max_value=0.1, value=0.003, step=0.001, format="%.3f",
                           help="Five-point period perturbation quotient")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ddp = st.slider('Average Difference of Differences (Jitter:DDP))', 
                           min_value=0.001, max_value=0.2, value=0.006, step=0.001, format="%.3f",
                           help="Average absolute difference of differences between jitter cycles")
        
        st.subheader("Amplitude Variations")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            shimmer = st.slider('Amplitude Variation (MDVP:Shimmer))', 
                              min_value=0.01, max_value=0.5, value=0.03, step=0.01, format="%.2f",
                              help="Variation in amplitude")
        with col2:
            shimmer_db = st.slider('Amplitude Variation in dB (MDVP:Shimmer(dB)))', 
                                min_value=0.1, max_value=2.0, value=0.25, step=0.01, format="%.2f",
                                help="Shimmer in decibels")
        with col3:
            apq3 = st.slider('3-Point Amplitude Perturbation (Shimmer:APQ3))', 
                            min_value=0.01, max_value=0.2, value=0.02, step=0.01, format="%.2f",
                            help="Three-point amplitude perturbation quotient")
        with col4:
            apq5 = st.slider('5-Point Amplitude Perturbation (Shimmer:APQ5))', 
                            min_value=0.01, max_value=0.2, value=0.02, step=0.01, format="%.2f",
                            help="Five-point amplitude perturbation quotient")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            apq = st.slider('Amplitude Perturbation Quotient (MDVP:APQ))', 
                           min_value=0.01, max_value=0.2, value=0.02, step=0.01, format="%.2f",
                           help="Amplitude perturbation quotient")
        with col2:
            dda = st.slider('Difference Between Consecutive Differences (Shimmer:DDA))', 
                           min_value=0.01, max_value=0.2, value=0.04, step=0.01, format="%.2f",
                           help="Average absolute difference between consecutive differences")
        
        st.subheader("Advanced Voice Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col3:
            nhr = st.slider('Noise-to-Harmonics Ratio (NHR))', 
                           min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f",
                           help="Ratio of noise to tonal components in the voice")
        with col4:
            hnr = st.slider('Harmonics-to-Noise Ratio (HNR))', 
                           min_value=0.0, max_value=40.0, value=22.0, step=0.1, format="%.1f",
                           help="Ratio of harmonic to noise components in dB")
        
        st.subheader("Nonlinear Measures")
        col1, col2, col3 = st.columns(3)
        with col1:
            rpde = st.slider('Recurrence Period Density Entropy (RPDE))', 
                            min_value=0.1, max_value=1.0, value=0.5, step=0.01, format="%.2f",
                            help="Measure of signal complexity")
        with col2:
            dfa = st.slider('Detrended Fluctuation Analysis (DFA))', 
                           min_value=0.1, max_value=1.5, value=0.7, step=0.01, format="%.2f",
                           help="Ratio of low to high frequency fluctuations")
        with col3:
            spread1 = st.slider('Nonlinear Measure of Frequency Variation (spread1))', 
                               min_value=-10.0, max_value=0.0, value=-5.0, step=0.1, format="%.1f",
                               help="Nonlinear measure of fundamental frequency variation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            spread2 = st.slider('Nonlinear Measure of Frequency Variation (spread2))', 
                              min_value=0.0, max_value=0.5, value=0.2, step=0.01, format="%.2f",
                              help="Nonlinear measure of fundamental frequency variation")
        with col2:
            d2 = st.slider('Correlation Dimension (D2))', 
                          min_value=1.0, max_value=5.0, value=2.3, step=0.1, format="%.1f",
                          help="Measure of signal complexity")
        with col3:
            ppe = st.slider('Pitch Period Entropy (PPE))', 
                           min_value=0.0, max_value=0.5, value=0.2, step=0.01, format="%.2f",
                           help="Measure of pitch period variation")

        submitted = st.form_submit_button("Assess Parkinson's Risk", type="primary", use_container_width=True)
        
        if submitted:
            input_data = {
                'MDVP:Fo(Hz)': fo,
                'MDVP:Fhi(Hz)': fhi,
                'MDVP:Flo(Hz)': flo,
                'MDVP:Jitter(%)': jitter_percent,
                'MDVP:Jitter(Abs)': jitter_abs,
                'MDVP:RAP': rap,
                'MDVP:PPQ': ppq,
                'Jitter:DDP': ddp,
                'MDVP:Shimmer': shimmer,
                'MDVP:Shimmer(dB)': shimmer_db,
                'Shimmer:APQ3': apq3,
                'Shimmer:APQ5': apq5,
                'MDVP:APQ': apq,
                'Shimmer:DDA': dda,
                'NHR': nhr,
                'HNR': hnr,
                'RPDE': rpde,
                'DFA': dfa,
                'spread1': spread1,
                'spread2': spread2,
                'D2': d2,
                'PPE': ppe
            }
            
            if not validate_parkinsons_inputs(input_data):
                return
                
            try:
                if len(input_data) != 22:
                    st.error(f"Error: Expected 22 features, got {len(input_data)}")
                    return
                    
                input_array = np.asarray(list(input_data.values())).reshape(1, -1)
                prediction = model.predict(input_array)
                proba = model.predict_proba(input_array)
                
                show_prediction_result(
                    prediction, proba,
                    "Potential Parkinson's disease indicators detected. Neurological consultation recommended for comprehensive evaluation.",
                    "No significant Parkinson's disease indicators detected in voice analysis."
                )
                
            except Exception as e:
                st.error(f'Error in prediction: {str(e)}')
                logging.error(f"Parkinson's prediction error: {str(e)}")
