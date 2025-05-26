import streamlit as st
import pandas as pd
import joblib
import os
import shap

# Load model
model_path = os.path.abspath("best_model.pkl")
model = joblib.load(model_path)

# Variable configuration
VARIABLE_CONFIG = {
    "Sex": {
        "min": 0, 
        "max": 1,
        "description": "Patient gender (0=Female, 1=Male)"
    },
    "ASA scores": {
        "min": 0,
        "max": 5,
        "description": "ASA physical status classification"
    },
    "tumor location": {
        "min": 1,
        "max": 4,
        "description": "Tumor location code (1-4)"
    },
    "Benign or malignant": {
        "min": 0,
        "max": 1,
        "description": "Tumor nature (0=Benign, 1=Malignant)"
    },
    "Admitted to NICU": {
        "min": 0,
        "max": 1,
        "description": "NICU admission status"
    },
    "Duration of surgery": {
        "min": 0,
        "max": 1,
        "description": "Surgery duration category"
    },
    "diabetes": {
        "min": 0,
        "max": 1,
        "description": "Diabetes mellitus status"
    },
    "CHF": {
        "min": 0,
        "max": 1,
        "description": "Congestive heart failure"
    },
    "Functional dependencies": {
        "min": 0,
        "max": 1,
        "description": "Functional dependencies"
    },
    "mFI-5": {
        "min": 0,
        "max": 5,
        "description": "Modified Frailty Index"
    },
    "Type of tumor": {
        "min": 1,
        "max": 5,
        "description": "Tumor type code (1-5)"
    }
}

# Interface setup
st.set_page_config(page_title="Surgical Risk System", layout="wide")
st.title("Unplanned Reoperation Risk Prediction System")
st.markdown("---")

# Dynamic input generation
inputs = {}
cols = st.columns(2)
for i, (feature, config) in enumerate(VARIABLE_CONFIG.items()):
    with cols[i % 2]:
        inputs[feature] = st.number_input(
            label=f"{feature} ({config['description']})",
            min_value=config["min"],
            max_value=config["max"],
            value=config["min"],
            step=1,
            key=feature
        )

# Prediction and explanation
if st.button("Predict Risk", type="primary"):
    try:
        # Create input dataframe
        input_df = pd.DataFrame([inputs])
        
        # Generate prediction
        proba = model.predict_proba(input_df)[0][1]
        risk_level = "High Risk" if proba > 0.5 else "Low Risk"
        color = "#FF4B4B" if proba > 0.5 else "#00CC96"
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{risk_level}</h2>", 
                    unsafe_allow_html=True)
        st.write(f"### Reoperation Probability: {proba:.1%}")
        
        # SHAP explanation
        st.markdown("---")
        st.subheader("Model Interpretation")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        with st.expander("Global Feature Importance"):
            fig_global = shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
            st.pyplot(fig_global)
            
        with st.expander("Individual Feature Impact"):
            fig_local = shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                input_df,
                matplotlib=True
            )
            st.pyplot(fig_local)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Sidebar documentation
with st.sidebar:
    st.markdown("""
    ## User Guide
    
    ### Input Requirements:
    1. All inputs should be integer values
    2. Numerical ranges are enforced for each parameter
    3. Default values set to minimum acceptable values
    
    ### Code Definitions:
    - **ASA scores**: 0-5 scale (0=Healthy, 5=Morbund)
    - **mFI-5**: 0-5 scale measuring frailty
    - **Tumor codes**: Numerical classification per institutional protocol
    """)