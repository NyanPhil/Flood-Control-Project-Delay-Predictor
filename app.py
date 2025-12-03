import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="DPWH On-Time Prediction",
    page_icon=":material/timeline:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# LOAD MODEL & PREPROCESSORS
# ============================================================================

@st.cache_resource
def load_model_and_preprocessors():
    """Load trained model and preprocessing objects."""
    model = joblib.load('random_forest_model.joblib')
    imputer_numeric = joblib.load('imputer_numeric.joblib')
    encoder = joblib.load('encoder.joblib')
    return model, imputer_numeric, encoder


model, imputer_numeric, encoder = load_model_and_preprocessors()


# ============================================================================
# DARK MODE STYLING
# ============================================================================

def load_dark_mode_css():
    """Apply dark mode CSS with developer-friendly colors."""
    css = """
    <style>
    /* Info and Status Cards */
    .info-card {
        background: linear-gradient(135deg, #1d2d3d 0%, #161b22 100%);
        border-left: 4px solid #a855f7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .info-card-text {
        color: #f0f6fc;
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 0.5rem;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid #30363d;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    
    .result-success {
        background: linear-gradient(135deg, #1a3a2a 0%, #0d1917 100%);
        border-left: 4px solid #3fb950;
        border-color: #30363d;
    }
    
    .result-delayed {
        background: linear-gradient(135deg, #3a1a1a 0%, #1a0d0d 100%);
        border-left: 4px solid #f85149;
        border-color: #30363d;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #f0f6fc;
    }
    
    .result-subtitle {
        font-size: 1.1rem;
        margin: 0.75rem 0 0 0;
        color: #8b949e;
    }
    
    .result-confidence {
        font-size: 0.95rem;
        margin: 1rem 0 0 0;
        color: #8b949e;
    }
    
    /* Input Section Headers */
    .section-header {
        color: #a855f7;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


load_dark_mode_css()

# ============================================================================
# DEFINE FEATURES (must match training)
# ============================================================================

numerical_features = [
    'ApprovedBudgetForContract', 'ContractCost', 'ProjectDuration',
    'AverageDuration_Region', 'AverageDuration_TypeOfWork', 'ContractorCount',
    'FundingYear', 'ProjectLatitude', 'ProjectLongitude',
    'ProvincialCapitalLatitude', 'ProvincialCapitalLongitude'
]

categorical_features = [
    'MainIsland', 'Region', 'Province', 'LegislativeDistrict',
    'Municipality', 'DistrictEngineeringOffice', 'TypeOfWork', 'ProvincialCapital'
]

# Get encoded feature names from the encoder
encoded_feature_names_from_dummy = []
if hasattr(encoder, 'get_feature_names_out'):
    try:
        encoded_feature_names_from_dummy = list(encoder.get_feature_names_out(categorical_features))
    except Exception as e:
        # Fallback if get_feature_names_out fails
        for cat_list, feature_name in zip(encoder.categories_, categorical_features):
            encoded_feature_names_from_dummy.extend([f"{feature_name}_{cat}" for cat in cat_list])

all_training_features = numerical_features + encoded_feature_names_from_dummy


# ============================================================================
# PAGE LAYOUT & HEADER
# ============================================================================

# Title and Description
st.markdown("# :material/timeline: DPWH Flood Control Project Prediction")
st.markdown("""
<div class="info-card">
    <p class="info-card-text">
        Predict whether a flood control project will complete on time (≤ median duration) 
        based on project characteristics and location data.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# INPUT FORMS - NUMERICAL FEATURES
# ============================================================================

st.markdown("## :material/calculate: Numerical Features")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("Budget & Cost", divider=False)
        approved_budget = st.number_input(
            'Approved Budget (PHP)', 
            min_value=0.0, 
            value=50000000.0, 
            step=1000000.0,
            help="Total approved contract budget in Philippine Pesos"
        )
        contract_cost = st.number_input(
            'Contract Cost (PHP)', 
            min_value=0.0, 
            value=48000000.0, 
            step=1000000.0,
            help="Actual contract cost in Philippine Pesos"
        )
        project_duration = st.number_input(
            'Duration (days)', 
            min_value=0, 
            value=150, 
            step=10,
            help="Expected project duration in days"
        )

with col2:
    with st.container(border=True):
        st.subheader("Project Details", divider=False)
        contractor_count = st.selectbox(
            'Number of Contractors', 
            options=[1, 2, 3], 
            index=0,
            help="Number of contractors involved"
        )
        funding_year = st.selectbox(
            'Funding Year', 
            options=list(range(2018, 2026)), 
            index=6,
            help="Year the project was funded"
        )
        project_latitude = st.number_input(
            'Project Latitude', 
            min_value=5.0, 
            max_value=20.0, 
            value=14.5, 
            format="%.6f",
            help="Geographic latitude of project"
        )

with col3:
    with st.container(border=True):
        st.subheader("Location Data", divider=False)
        project_longitude = st.number_input(
            'Project Longitude', 
            min_value=117.0, 
            max_value=127.0, 
            value=121.0, 
            format="%.6f",
            help="Geographic longitude of project"
        )
        avg_duration_region = st.number_input(
            'Avg Duration (Region)', 
            min_value=0.0, 
            value=250.0,
            help="Average project duration in this region"
        )
        avg_duration_typeofwork = st.number_input(
            'Avg Duration (Work Type)', 
            min_value=0.0, 
            value=250.0,
            help="Average duration for this type of work"
        )
        prov_capital_latitude = st.number_input(
            'Capital Latitude', 
            min_value=5.0, 
            max_value=20.0, 
            value=14.0, 
            format="%.6f",
            help="Latitude of provincial capital"
        )
        prov_capital_longitude = st.number_input(
            'Capital Longitude', 
            min_value=117.0, 
            max_value=127.0, 
            value=121.0, 
            format="%.6f",
            help="Longitude of provincial capital"
        )


# ============================================================================
# INPUT FORMS - CATEGORICAL FEATURES
# ============================================================================

st.markdown("## :material/category: Categorical Features")

col_cat1, col_cat2 = st.columns(2)

with col_cat1:
    with st.container(border=True):
        st.subheader("Geographic", divider=False)
        input_mainisland = st.selectbox(
            'Main Island', 
            encoder.categories_[0],
            help="Primary island where project is located"
        )
        input_region = st.selectbox(
            'Region', 
            encoder.categories_[1],
            help="Regional administrative division"
        )
        input_province = st.selectbox(
            'Province', 
            encoder.categories_[2],
            help="Provincial location"
        )

with col_cat2:
    with st.container(border=True):
        st.subheader("Administrative", divider=False)
        input_legislativedistrict = st.selectbox(
            'Legislative District', 
            encoder.categories_[3],
            help="Legislative district"
        )
        input_municipality = st.selectbox(
            'Municipality', 
            encoder.categories_[4],
            help="Municipal location"
        )
        input_deo = st.selectbox(
            'District Engineering Office', 
            encoder.categories_[5],
            help="Responsible engineering office"
        )

col_cat3 = st.columns(1)[0]
with col_cat3:
    with st.container(border=True):
        st.subheader("Project Type", divider=False)
        input_typeofwork = st.selectbox(
            'Type Of Work', 
            encoder.categories_[6],
            help="Category of flood control work"
        )
        input_provcapital = st.selectbox(
            'Provincial Capital', 
            encoder.categories_[7],
            help="Nearest provincial capital"
        )


# ============================================================================
# PREDICTION
# ============================================================================

col_predict_left, col_predict_center, col_predict_right = st.columns([1, 1, 1])

with col_predict_center:
    predict_button = st.button(
        ':material/smart_toy: Make Prediction',
        use_container_width=True,
        type="primary",
        help="Click to predict project completion status"
    )


if predict_button:
    # Create input dictionary
    input_data_dict = {
        'ApprovedBudgetForContract': approved_budget,
        'ContractCost': contract_cost,
        'ProjectDuration': float(project_duration),
        'AverageDuration_Region': avg_duration_region,
        'AverageDuration_TypeOfWork': avg_duration_typeofwork,
        'ContractorCount': float(contractor_count),
        'FundingYear': float(funding_year),
        'ProjectLatitude': project_latitude,
        'ProjectLongitude': project_longitude,
        'ProvincialCapitalLatitude': prov_capital_latitude,
        'ProvincialCapitalLongitude': prov_capital_longitude,
        'MainIsland': input_mainisland,
        'Region': input_region,
        'Province': input_province,
        'LegislativeDistrict': input_legislativedistrict,
        'Municipality': input_municipality,
        'DistrictEngineeringOffice': input_deo,
        'TypeOfWork': input_typeofwork,
        'ProvincialCapital': input_provcapital
    }

    # Convert to DataFrame
    new_df = pd.DataFrame([input_data_dict])

    # Preprocess: Impute numerical features
    new_df[numerical_features] = imputer_numeric.transform(new_df[numerical_features])

    # Preprocess: One-hot encode categorical features
    encoded_new_features = encoder.transform(new_df[categorical_features])
    encoded_new_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_new_df = pd.DataFrame(
        encoded_new_features, 
        columns=encoded_new_feature_names, 
        index=new_df.index
    )

    # Combine numerical and encoded features
    X_new = pd.concat([new_df[numerical_features], encoded_new_df], axis=1)

    # Align columns with training data
    missing_cols = set(all_training_features) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0

    X_new = X_new[all_training_features]

    # Make prediction
    prediction = model.predict(X_new)
    prediction_proba = model.predict_proba(X_new)

    # Display results
    st.markdown("## :material/task_alt: Prediction Results")

    if prediction[0] == 1:
        # On-time prediction
        confidence = prediction_proba[0][1] * 100
        st.markdown(f"""
        <div class="result-card result-success">
            <p class="result-title">ON TIME</p>
            <p class="result-subtitle">Project will likely complete on schedule</p>
            <p class="result-confidence"><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional context
        col_context_1, col_context_2 = st.columns(2)
        with col_context_1:
            st.success("✓ Project looks well-positioned for timely completion")
        with col_context_2:
            st.metric("Probability of On-Time Completion", f"{confidence:.1f}%")
    else:
        # Delayed prediction
        confidence = prediction_proba[0][0] * 100
        st.markdown(f"""
        <div class="result-card result-delayed">
            <p class="result-title">DELAYED</p>
            <p class="result-subtitle">Project may face delays</p>
            <p class="result-confidence"><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional context
        col_context_1, col_context_2 = st.columns(2)
        with col_context_1:
            st.warning("⚠ Project may experience delays - monitor closely")
        with col_context_2:
            st.metric("Probability of Delay", f"{confidence:.1f}%")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #8b949e; padding: 2rem 0;">
    <p><strong>DPWH Flood Control Project On-Time Prediction System</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)


