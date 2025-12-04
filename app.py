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

# Initialize session state for tab selection
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0


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
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_process_training_data():
    """Load the actual training data to compute averages for engineering features."""
    try:
        df = pd.read_csv('data/dpwh_flood_control_projects.csv')
        
        # Calculate ProjectDuration
        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df['ActualCompletionDate'] = pd.to_datetime(df['ActualCompletionDate'])
        df['ProjectDuration'] = (df['ActualCompletionDate'] - df['StartDate']).dt.days
        
        # Calculate average durations by Region and TypeOfWork
        avg_by_region = df.groupby('Region')['ProjectDuration'].mean().to_dict()
        avg_by_work_type = df.groupby('TypeOfWork')['ProjectDuration'].mean().to_dict()
        
        return avg_by_region, avg_by_work_type
    except Exception as e:
        st.warning(f"Could not load training data for feature engineering: {str(e)}")
        return {}, {}


avg_duration_by_region, avg_duration_by_work_type = load_and_process_training_data()


def engineer_features(user_data: dict, avg_by_region: dict, avg_by_work_type: dict, global_avg: float = 250.0) -> dict:
    """
    Engineer the missing features from raw user inputs.
    
    Parameters:
    user_data: Dictionary with raw user inputs
    avg_by_region: Dictionary of average durations by region
    avg_by_work_type: Dictionary of average durations by work type
    global_avg: Global average duration fallback
    
    Returns:
    Dictionary with all features including engineered ones
    """
    engineered_data = user_data.copy()
    
    # Ensure all numerical fields are properly converted to float
    numerical_fields = ['ApprovedBudgetForContract', 'ContractCost', 'ContractorCount', 
                        'FundingYear', 'ProjectLatitude', 'ProjectLongitude',
                        'ProvincialCapitalLatitude', 'ProvincialCapitalLongitude']
    
    for field in numerical_fields:
        if field in engineered_data:
            try:
                engineered_data[field] = float(engineered_data[field])
            except (ValueError, TypeError):
                engineered_data[field] = None
    
    # Engineer ProjectDuration from dates
    if 'StartDate' in user_data and 'ActualCompletionDate' in user_data:
        try:
            start = pd.to_datetime(user_data['StartDate'])
            end = pd.to_datetime(user_data['ActualCompletionDate'])
            duration = float((end - start).days)
            # Ensure duration is not negative (take absolute value if dates are reversed)
            engineered_data['ProjectDuration'] = abs(duration)
        except:
            engineered_data['ProjectDuration'] = 150.0  # Default fallback
    
    # Engineer AverageDuration_Region
    region = user_data.get('Region', '')
    if region in avg_by_region:
        engineered_data['AverageDuration_Region'] = float(avg_by_region[region])
    else:
        engineered_data['AverageDuration_Region'] = float(global_avg)
    
    # Engineer AverageDuration_TypeOfWork
    work_type = user_data.get('TypeOfWork', '')
    if work_type in avg_by_work_type:
        engineered_data['AverageDuration_TypeOfWork'] = float(avg_by_work_type[work_type])
    else:
        engineered_data['AverageDuration_TypeOfWork'] = float(global_avg)
    
    return engineered_data


# ============================================================================
# DARK MODE STYLING
# ============================================================================

def load_dark_mode_css():
    """Apply dark mode CSS with developer-friendly colors."""
    css = """
    <style>
    /* Font Face Declarations - Using Streamlit static file serving */
    /* Streamlit serves files from the 'static' folder at the root */
    /* Using multiple path formats for compatibility */
    @font-face {
        font-family: 'Inter';
        src: url('/static/Inter_18pt-Regular.ttf') format('truetype'),
             url('static/Inter_18pt-Regular.ttf') format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'Inter';
        src: url('/static/Inter_18pt-Medium.ttf') format('truetype'),
             url('static/Inter_18pt-Medium.ttf') format('truetype');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'Inter';
        src: url('/static/Inter_18pt-SemiBold.ttf') format('truetype'),
             url('static/Inter_18pt-SemiBold.ttf') format('truetype');
        font-weight: 600;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'Inter';
        src: url('/static/Inter_18pt-Bold.ttf') format('truetype'),
             url('static/Inter_18pt-Bold.ttf') format('truetype');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'JetBrains Mono';
        src: url('/static/JetBrainsMono-Regular.ttf') format('truetype'),
             url('static/JetBrainsMono-Regular.ttf') format('truetype');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'JetBrains Mono';
        src: url('/static/JetBrainsMono-Medium.ttf') format('truetype'),
             url('static/JetBrainsMono-Medium.ttf') format('truetype');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'JetBrains Mono';
        src: url('/static/JetBrainsMono-Bold.ttf') format('truetype'),
             url('static/JetBrainsMono-Bold.ttf') format('truetype');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }
    
    /* Apply Inter font with !important to override Streamlit defaults */
    /* Emoji fonts in fallback ensure icons render correctly */
    body, .stApp, .main {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Apply to all text content */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
    .stMarkdown div, .stText, .stCaption, .stSubheader, .stHeader {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Headings with emoji support */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Form elements */
    input[type="text"], input[type="number"], input[type="date"],
    select, textarea, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Streamlit widget text */
    .stSelectbox, .stSelectbox label, .stNumberInput, .stNumberInput label, 
    .stTextInput, .stTextInput label, .stDateInput, .stDateInput label,
    .stSelectbox [data-baseweb="select"], .stSelectbox [data-baseweb="select"] span,
    .stNumberInput input, .stTextInput input {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Buttons */
    button, .stButton button {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Apply JetBrains Mono to code */
    code, pre, .stCodeBlock, .stCodeBlock code, .stMarkdownContainer code {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    
    /* Custom card text */
    .info-card, .info-card-text, .result-card, .result-title, 
    .result-subtitle, .result-confidence {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Ensure emoji characters use emoji fonts (they will automatically due to font fallback) */
    /* The emoji fonts in the fallback chain will handle emoji/icon rendering */
    
    /* Info and Status Cards */
    .info-card {
        background: linear-gradient(135deg, #1d2d3d 0%, #161b22 100%);
        border-left: 4px solid #a855f7;
        border-top: 1px solid rgba(168, 85, 247, 0.3);
        border-right: 1px solid rgba(168, 85, 247, 0.2);
        border-bottom: 1px solid rgba(168, 85, 247, 0.2);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.2), 0 0 20px rgba(168, 85, 247, 0.1);
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
        border-top: 1px solid rgba(63, 185, 80, 0.3);
        border-right: 1px solid rgba(63, 185, 80, 0.2);
        border-bottom: 1px solid rgba(63, 185, 80, 0.2);
        box-shadow: 0 8px 24px rgba(63, 185, 80, 0.2), 0 0 30px rgba(63, 185, 80, 0.1);
    }
    
    .result-delayed {
        background: linear-gradient(135deg, #3a1a1a 0%, #1a0d0d 100%);
        border-left: 4px solid #f85149;
        border-top: 1px solid rgba(248, 81, 73, 0.3);
        border-right: 1px solid rgba(248, 81, 73, 0.2);
        border-bottom: 1px solid rgba(248, 81, 73, 0.2);
        box-shadow: 0 8px 24px rgba(248, 81, 73, 0.2), 0 0 30px rgba(248, 81, 73, 0.1);
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
    
    /* Colorful accents for Streamlit containers */
    [data-testid="stVerticalBlock"] > [style*="border"] {
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.4) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #9333ea 0%, #7e22ce 100%) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Colorful input field accents */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.2) !important;
    }
    
    /* Colorful section containers with better spacing */
    .stContainer {
        border-left: 3px solid rgba(168, 85, 247, 0.5) !important;
        padding: 1.5rem !important;
        padding-left: 2rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Better spacing for form sections */
    [data-testid="stVerticalBlock"] {
        gap: 1.5rem !important;
    }
    
    /* Spacing for input groups */
    .stNumberInput, .stTextInput, .stSelectbox, .stDateInput {
        margin-bottom: 1.5rem !important;
    }
    
    /* Spacing for columns */
    [data-testid="column"] {
        padding: 0 1rem !important;
    }
    
    /* Better spacing for markdown sections */
    .stMarkdown {
        margin: 1.5rem 0 !important;
    }
    
    .stMarkdown p {
        margin: 1rem 0 !important;
        line-height: 1.6 !important;
    }
    
    /* Accent colors for section headers - using general h2 styling */
    h2 {
        border-bottom: 2px solid rgba(168, 85, 247, 0.3) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Colorful markdown headers */
    .stMarkdown h2 {
        color: #a855f7 !important;
        border-bottom: 2px solid rgba(168, 85, 247, 0.4) !important;
    }
    
    .stMarkdown h3 {
        color: #60a5fa !important;
    }
    
    /* Colorful metric displays */
    [data-testid="stMetricValue"] {
        color: #a855f7 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    
    /* Success and warning messages with colors */
    .stSuccess {
        background: linear-gradient(135deg, rgba(63, 185, 80, 0.1) 0%, rgba(63, 185, 80, 0.05) 100%) !important;
        border-left: 4px solid #3fb950 !important;
        border-radius: 0.5rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 0.5rem !important;
    }
    
    /* Colorful tabs/radio buttons */
    .stRadio > div {
        background: rgba(168, 85, 247, 0.1) !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
    }
    
    /* Radio button labels with custom font */
    .stRadio label,
    .stRadio label[data-testid="stRadioLabel"],
    .stRadio span,
    .stRadio > div > div > label,
    .stRadio > div > div > span,
    .stRadio * {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Exclude radio input circles from font override */
    .stRadio input[type="radio"] {
        font-family: initial !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }
    
    /* Colorful dataframe styling */
    .stDataFrame {
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 0.5rem !important;
        overflow: hidden !important;
    }
    
    /* Table header accents */
    table thead th {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(168, 85, 247, 0.1) 100%) !important;
        color: #a855f7 !important;
        font-weight: 600 !important;
    }
    
    /* Alternating row colors with subtle accent */
    table tbody tr:nth-child(even) {
        background: rgba(168, 85, 247, 0.05) !important;
    }
    
    /* File uploader styling with better spacing */
    .stFileUploader {
        border: 2px dashed rgba(168, 85, 247, 0.4) !important;
        border-radius: 0.5rem !important;
        background: rgba(168, 85, 247, 0.05) !important;
        padding: 2rem !important;
        margin: 0.5rem 0 1.5rem 0 !important;
    }
    
    .stFileUploader > div {
        padding: 1.5rem !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        padding: 3rem 2rem !important;
        min-height: 150px !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        padding: 1rem 0 !important;
        margin: 1rem 0 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] > div {
        margin: 0.75rem 0 !important;
    }
    
    .stFileUploader button {
        margin-top: 1.5rem !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .stFileUploader:hover {
        border-color: rgba(168, 85, 247, 0.6) !important;
        background: rgba(168, 85, 247, 0.1) !important;
    }
    
    /* File uploader label and help text spacing */
    .stFileUploader label {
        margin-bottom: 1rem !important;
        padding-bottom: 0.75rem !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderLabel"] {
        margin-bottom: 1.5rem !important;
        padding: 0.5rem 0 !important;
    }
    
    /* Spacing for file uploader help icon */
    .stFileUploader [data-testid="stTooltipIcon"] {
        margin-left: 0.5rem !important;
    }
    
    /* Better spacing for the drag and drop area content */
    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] p {
        margin: 0.5rem 0 !important;
        line-height: 1.8 !important;
    }
    
    /* Date input styling */
    .stDateInput > div > div {
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    
    .stDateInput > div > div:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.2) !important;
    }
    
    /* Caption and subheader colors */
    .stCaption {
        color: #8b949e !important;
    }
    
    /* Divider with accent color */
    hr {
        border-color: rgba(168, 85, 247, 0.3) !important;
        margin: 2rem 0 !important;
    }
    
    /* Info box accent */
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 0.5rem !important;
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.1) 0%, rgba(248, 81, 73, 0.05) 100%) !important;
        border-left: 4px solid #f85149 !important;
        border-radius: 0.5rem !important;
    }
    
    /* Main title styling with gradient */
    h1, .stMarkdown h1 {
        background: linear-gradient(135deg, #a855f7 0%, #60a5fa 50%, #34d399 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
        display: inline-block !important;
    }
    
    /* Fallback for browsers that don't support text gradient */
    @supports not (-webkit-background-clip: text) {
        h1, .stMarkdown h1 {
            color: #a855f7 !important;
            -webkit-text-fill-color: #a855f7 !important;
        }
    }
    
    /* Sidebar accent */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        border-right: 2px solid rgba(168, 85, 247, 0.3) !important;
    }
    
    /* Main content area subtle accent */
    .main .block-container {
        background: linear-gradient(180deg, transparent 0%, rgba(168, 85, 247, 0.02) 100%) !important;
    }
    
    /* Selectbox dropdown accent */
    [data-baseweb="select"] {
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    
    [data-baseweb="select"]:hover {
        border-color: rgba(168, 85, 247, 0.5) !important;
    }
    
    /* Number input spinner accent */
    .stNumberInput button {
        color: #a855f7 !important;
        border-color: rgba(168, 85, 247, 0.3) !important;
    }
    
    .stNumberInput button:hover {
        background: rgba(168, 85, 247, 0.1) !important;
        border-color: #a855f7 !important;
    }
    
    /* Radio button selected state */
    .stRadio label[data-testid="stRadioLabel"]:has(input:checked) {
        color: #a855f7 !important;
        font-weight: 600 !important;
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Ensure all radio button text uses Inter font */
    .stRadio input + span,
    .stRadio input + label {
        font-family: 'Inter', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
                     'Noto Color Emoji', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Container borders with gradient effect */
    [data-testid="stVerticalBlock"] > [style*="border"] {
        border-image: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(96, 165, 250, 0.3)) 1 !important;
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

# Header with Image and Title
st.markdown("""
<div style="position: relative; margin-bottom: 2rem; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 8px 32px rgba(168, 85, 247, 0.2);">
    <img src="app/static/Dams.jpg" style="width: 100%; height: 300px; object-fit: cover; display: block;">
    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(135deg, rgba(13, 17, 23, 0.7) 0%, rgba(13, 17, 23, 0.5) 100%); display: flex; align-items: center; justify-content: center;">
        <div style="text-align: center; color: white;">
            <h1 style="font-size: 2.5rem; font-weight: 800; margin: 0; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);">
                ⏱️ DPWH Flood Control Project Delay Predictor
            </h1>
            <p style="font-size: 1.1rem; margin: 0.75rem 0 0 0; color: #e0e0e0; text-shadow: 0 1px 4px rgba(0, 0, 0, 0.5);">
                Intelligent prediction system for project completion timelines
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Description Card
st.markdown("""
<div class="info-card">
    <p class="info-card-text">
        Predict whether a flood control project will complete on time (≤ median duration) 
        based on project characteristics and location data.
    </p>
</div>
""", unsafe_allow_html=True)

# Create tabs for single and batch predictions
# Check if user just uploaded a file - if so, stay on batch upload tab
default_tab = "Batch Upload" if st.session_state.get('batch_uploaded', False) else "Single Project"

selected_tab = st.radio(
    "Navigation", 
    ["Single Project", "Batch Upload"], 
    horizontal=True, 
    label_visibility="collapsed",
    index=0 if default_tab == "Single Project" else 1
)

# Reset the flag after using it
if selected_tab == "Single Project":
    st.session_state.batch_uploaded = False

if selected_tab == "Batch Upload":
    st.markdown("## :material/upload_file: Upload Project List")
    st.markdown("Upload a CSV file with project data to conduct predictions on multiple projects at once.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should contain columns matching the input features"
    )
    
    if uploaded_file is not None:
        # Store that we're on batch upload tab
        st.session_state.batch_uploaded = True
        try:
            # Read the CSV file
            df_uploaded = pd.read_csv(uploaded_file)

            # Defensive conversion: replace 'nan' strings and empty strings with pd.NA
            df_uploaded.replace({'nan': pd.NA, '': pd.NA}, inplace=True)

            # List of all numeric columns expected in raw input
            numeric_cols = [
                'ApprovedBudgetForContract', 'ContractCost', 'ContractorCount', 
                'FundingYear', 'ProjectLatitude', 'ProjectLongitude',
                'ProvincialCapitalLatitude', 'ProvincialCapitalLongitude'
            ]
            for col in numeric_cols:
                if col in df_uploaded.columns:
                    # Remove whitespace and ensure proper conversion
                    df_uploaded[col] = df_uploaded[col].astype(str).str.strip().replace({'nan': pd.NA, '': pd.NA})
                    df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')
            # After conversion, check for any remaining string types and coerce
            for col in numeric_cols:
                if col in df_uploaded.columns and df_uploaded[col].dtype == object:
                    df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')

            # Validate that required raw columns exist
            required_raw_cols = [
                'ProjectName', 'ApprovedBudgetForContract', 'ContractCost',
                'StartDate', 'ActualCompletionDate', 'ContractorCount', 'FundingYear',
                'ProjectLatitude', 'ProjectLongitude', 
                'ProvincialCapitalLatitude', 'ProvincialCapitalLongitude',
                'MainIsland', 'Region', 'Province', 'LegislativeDistrict',
                'Municipality', 'DistrictEngineeringOffice', 'TypeOfWork', 'ProvincialCapital'
            ]
            missing_cols = [col for col in required_raw_cols if col not in df_uploaded.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success(f"✓ File loaded successfully with {len(df_uploaded)} projects")
                
                # Convert date columns
                df_uploaded['StartDate'] = pd.to_datetime(df_uploaded['StartDate'])
                df_uploaded['ActualCompletionDate'] = pd.to_datetime(df_uploaded['ActualCompletionDate'])
                
                # Convert all numeric columns to proper types
                numeric_cols = ['ApprovedBudgetForContract', 'ContractCost', 'ContractorCount', 
                               'FundingYear', 'ProjectLatitude', 'ProjectLongitude',
                               'ProvincialCapitalLatitude', 'ProvincialCapitalLongitude']
                for col in numeric_cols:
                    if col in df_uploaded.columns:
                        df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')
                
                # Prepare predictions for all projects
                predictions_list = []
                
                for idx, row in df_uploaded.iterrows():
                    # Create raw input data - ensure all numeric values are float
                    raw_input_data = {
                        'ApprovedBudgetForContract': float(row['ApprovedBudgetForContract']),
                        'ContractCost': float(row['ContractCost']),
                        'StartDate': row['StartDate'],
                        'ActualCompletionDate': row['ActualCompletionDate'],
                        'ContractorCount': float(row['ContractorCount']),
                        'FundingYear': float(row['FundingYear']),
                        'ProjectLatitude': float(row['ProjectLatitude']),
                        'ProjectLongitude': float(row['ProjectLongitude']),
                        'ProvincialCapitalLatitude': float(row['ProvincialCapitalLatitude']),
                        'ProvincialCapitalLongitude': float(row['ProvincialCapitalLongitude']),
                        'MainIsland': row['MainIsland'],
                        'Region': row['Region'],
                        'Province': row['Province'],
                        'LegislativeDistrict': row['LegislativeDistrict'],
                        'Municipality': row['Municipality'],
                        'DistrictEngineeringOffice': row['DistrictEngineeringOffice'],
                        'TypeOfWork': row['TypeOfWork'],
                        'ProvincialCapital': row['ProvincialCapital']
                    }
                    
                    # Engineer missing features
                    input_data_dict = engineer_features(raw_input_data, avg_duration_by_region, avg_duration_by_work_type)
                    
                    # Convert to DataFrame
                    new_df = pd.DataFrame([input_data_dict])
                    
                    # Ensure all numerical features are numeric type and handle NaN strings
                    for num_feat in numerical_features:
                        if num_feat in new_df.columns:
                            try:
                                # Replace 'nan' strings with actual NaN
                                if isinstance(new_df[num_feat].iloc[0], str):
                                    new_df[num_feat] = new_df[num_feat].replace('nan', pd.NA)
                                new_df[num_feat] = pd.to_numeric(new_df[num_feat], errors='coerce')
                            except:
                                pass
                    
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
                    
                    # Get project duration and contract cost
                    project_duration = input_data_dict.get('ProjectDuration', 0)
                    contract_cost = row['ContractCost']
                    
                    # Store result
                    predictions_list.append({
                        'Project Name': row['ProjectName'],
                        'Project Duration (days)': f"{int(project_duration)}",
                        'Contract Cost': f"₱{contract_cost:,.2f}",
                        'Status': 'On Time' if prediction[0] == 1 else 'Delayed',
                        'Confidence': f"{max(prediction_proba[0]) * 100:.1f}%"
                    })
                
                # Display results as table
                results_df = pd.DataFrame(predictions_list)
                
                st.markdown("### Prediction Results")
                
                # Color coding for status
                def color_status(val):
                    if val == 'On Time':
                        return 'background-color: #1a3a2a'
                    else:
                        return 'background-color: #3a1a1a'
                
                styled_df = results_df.style.map(color_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_results,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:  # Single Project tab

    # ============================================================================
    # INPUT FORMS - NUMERICAL FEATURES
    # ============================================================================

    st.markdown("## :material/calculate: Raw Project Data")
    st.caption("Enter the base project information. Duration and averages will be automatically calculated.")

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

    with col2:
        with st.container(border=True):
            st.subheader("Project Timeline", divider=False)
            start_date = st.date_input(
                'Start Date',
                value=pd.Timestamp('2022-01-01'),
                help="Project start date"
            )
            completion_date = st.date_input(
                'Completion Date',
                value=pd.Timestamp('2022-06-30'),
                help="Project completion date (for duration calculation)"
            )

    with col3:
        with st.container(border=True):
            st.subheader("Project Team", divider=False)
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


    # ============================================================================
    # INPUT FORMS - LOCATION FEATURES
    # ============================================================================

    st.markdown("## :material/location_on: Project Location")

    col_loc1, col_loc2 = st.columns(2)

    with col_loc1:
        with st.container(border=True):
            st.subheader("Project Coordinates", divider=False)
            project_latitude = st.number_input(
                'Project Latitude', 
                min_value=5.0, 
                max_value=20.0, 
                value=14.5, 
                format="%.6f",
                help="Geographic latitude of project location"
            )
            project_longitude = st.number_input(
                'Project Longitude', 
                min_value=117.0, 
                max_value=127.0, 
                value=121.0, 
                format="%.6f",
                help="Geographic longitude of project location"
            )

    with col_loc2:
        with st.container(border=True):
            st.subheader("Provincial Capital", divider=False)
            prov_capital_latitude = st.number_input(
                'Capital Latitude', 
                min_value=5.0, 
                max_value=20.0, 
                value=14.0, 
                format="%.6f",
                help="Latitude of nearest provincial capital"
            )
            prov_capital_longitude = st.number_input(
                'Capital Longitude', 
                min_value=117.0, 
                max_value=127.0, 
                value=121.0, 
                format="%.6f",
                help="Longitude of nearest provincial capital"
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
        # Create raw input dictionary (only non-engineered features)
        raw_input_data = {
            'ApprovedBudgetForContract': approved_budget,
            'ContractCost': contract_cost,
            'StartDate': start_date,
            'ActualCompletionDate': completion_date,
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

        # Engineer missing features
        input_data_dict = engineer_features(raw_input_data, avg_duration_by_region, avg_duration_by_work_type)

        # Convert to DataFrame
        new_df = pd.DataFrame([input_data_dict])

        # Ensure all numerical features are numeric type and handle NaN strings
        for num_feat in numerical_features:
            if num_feat in new_df.columns:
                try:
                    # Replace 'nan' strings with actual NaN
                    if isinstance(new_df[num_feat].iloc[0], str):
                        new_df[num_feat] = new_df[num_feat].replace('nan', pd.NA)
                    new_df[num_feat] = pd.to_numeric(new_df[num_feat], errors='coerce')
                except:
                    pass

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


