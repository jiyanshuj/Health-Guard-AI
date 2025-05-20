import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import Counter
from typing import Tuple, Optional
import logging
import streamlit as st
from config import MODEL_FILES

def safe_train_test_split(X: pd.DataFrame, 
                         Y: pd.Series, 
                         test_size: float = 0.2, 
                         random_state: int = 2) -> Tuple:
    """Safe train-test split that handles small classes."""
    class_counts = Counter(Y)
    if min(class_counts.values()) < 2:
        logging.warning("One class has fewer than 2 samples - using non-stratified split")
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=random_state)

def train_diabetes_model() -> Optional[Pipeline]:
    """Train and evaluate diabetes prediction model."""
    try:
        diabetes_data = pd.read_csv(
            'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv',
            header=None,
            names=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                   'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        )
        for col in diabetes_data.columns[:-1]:
            diabetes_data[col] = pd.to_numeric(diabetes_data[col], errors='coerce')
        diabetes_data = diabetes_data[
            (diabetes_data['Glucose'] >= 40) & (diabetes_data['Glucose'] <= 400) &
            (diabetes_data['BloodPressure'] >= 20) & (diabetes_data['BloodPressure'] <= 180) &
            (diabetes_data['BMI'] >= 10) & (diabetes_data['BMI'] <= 60)
        ].dropna()
        X = diabetes_data.drop(columns='Outcome', axis=1)
        Y = diabetes_data['Outcome']
        X_train, X_test, Y_train, Y_test = safe_train_test_split(X, Y)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='linear', probability=True, random_state=42))
        ])
        model.fit(X_train, Y_train)
        cv_scores = cross_val_score(model, X, Y, cv=5)
        logging.info(f"Diabetes Model CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
        pickle.dump(model, open(MODEL_FILES['diabetes'], 'wb'))
        return model
    except Exception as e:
        logging.error(f"Error training diabetes model: {str(e)}")
        st.error(f"Error training diabetes model: {str(e)}")
        return None

def train_heart_model() -> Optional[Pipeline]:
    """Train and evaluate heart disease prediction model."""
    try:
        heart_data = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            header=None,
            names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        )
        heart_data = heart_data.replace('?', np.nan)
        for col in heart_data.columns:
            heart_data[col] = pd.to_numeric(heart_data[col], errors='coerce')
        heart_data = heart_data[
            (heart_data['trestbps'] >= 50) & (heart_data['trestbps'] <= 250) &
            (heart_data['chol'] >= 100) & (heart_data['chol'] <= 600) &
            (heart_data['thalach'] >= 50) & (heart_data['thalach'] <= 250)
        ].dropna()
        heart_data['target'] = heart_data['target'].apply(lambda x: 0 if x == 0 else 1)
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']
        X_train, X_test, Y_train, Y_test = safe_train_test_split(X, Y)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        model.fit(X_train, Y_train)
        cv_scores = cross_val_score(model, X, Y, cv=5)
        logging.info(f"Heart Model CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
        pickle.dump(model, open(MODEL_FILES['heart'], 'wb'))
        return model
    except Exception as e:
        logging.error(f"Error training heart model: {str(e)}")
        st.error(f"Error training heart model: {str(e)}")
        return None

def train_parkinsons_model() -> Optional[Pipeline]:
    """Train and evaluate Parkinson's disease prediction model."""
    try:
        parkinsons_data = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        )
        for col in parkinsons_data.columns:
            if col not in ['name', 'status']:
                parkinsons_data[col] = pd.to_numeric(parkinsons_data[col], errors='coerce')
        parkinsons_data = parkinsons_data.dropna()
        X = parkinsons_data.drop(columns=['name','status'], axis=1)
        Y = parkinsons_data['status']
        X_train, X_test, Y_train, Y_test = safe_train_test_split(X, Y)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', svm.SVC(kernel='linear', probability=True, random_state=42))
        ])
        model.fit(X_train, Y_train)
        cv_scores = cross_val_score(model, X, Y, cv=5)
        logging.info(f"Parkinson's Model CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")
        pickle.dump(model, open(MODEL_FILES['parkinsons'], 'wb'))
        return model
    except Exception as e:
        logging.error(f"Error training Parkinson's model: {str(e)}")
        st.error(f"Error training Parkinson's model: {str(e)}")
        return None
