# HealthGuard AI

HealthGuard AI is a Streamlit-based web application that provides preliminary risk assessments for three major health conditions: Diabetes Mellitus, Coronary Heart Disease, and Parkinson's Disease. Using machine learning models trained on clinical datasets, the app offers risk estimates with confidence levels and personalized health recommendations.

---

## Features

- Risk prediction for:
  - Diabetes Mellitus
  - Coronary Heart Disease
  - Parkinson's Disease
- Interactive and user-friendly interface built with Streamlit
- Automatic model loading and training if models are not found locally
- Displays model status and training progress
- Feedback submission feature to help improve the app
- Clear disclaimers emphasizing the informational nature of the tool

---

## Installation

1. Ensure you have Python 3.7 or higher installed.
2. Clone this repository or download the source code.
3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## Usage

To run the HealthGuard AI app locally, execute the following command in the project directory:

```bash
streamlit run main.py
```

This will launch the app in your default web browser. Use the sidebar to navigate between Diabetes, Heart Disease, and Parkinson's risk assessment pages.

---

## Model Training

The app uses pre-trained machine learning models saved as `.sav` files. If these model files are not found, the app will automatically train the models using publicly available datasets:

- Diabetes model trained on the Pima Indians Diabetes dataset
- Heart disease model trained on the Cleveland Heart Disease dataset
- Parkinson's disease model trained on the Parkinson's dataset from the UCI Machine Learning Repository

Training progress is displayed within the app interface.

---

## Feedback

Users can provide feedback directly through the app via the feedback section. Feedback is logged and helps the development team improve HealthGuard AI.

---

## Disclaimer

HealthGuard AI is intended for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any medical concerns.

---

## Version and Authors

- Version: 1.1.0
- Developed by the HealthGuard AI Team
