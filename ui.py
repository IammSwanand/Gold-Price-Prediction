"""
UI Components and Styling
Handles Streamlit page configuration and styling
"""
import streamlit as st


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="🥇 Gold Price Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .signal-buy {
            background-color: #d4edda;
            color: #155724;
        }
        .signal-no {
            background-color: #fff3cd;
            color: #856404;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'model_trained': False,
        'train_triggered': False,
        'cached_model': None,
        'cached_data': None,
        'cached_model_type': 'Linear Regression',
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def sidebar_header():
    """Create sidebar header and common buttons"""
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")


def refresh_data_button():
    """Add refresh data cache button"""
    if st.sidebar.button(
        "🔄 Refresh Data Cache",
        use_container_width=True,
        help="Clear cached data and fetch latest from yfinance"
    ):
        st.cache_data.clear()
        st.rerun()


def train_model_button():
    """Add train model button"""
    if st.sidebar.button(
        "🚀 Train Model",
        use_container_width=True,
        help="Click to train/retrain the model with current parameters"
    ):
        st.session_state.train_triggered = True
    
    # Auto-train on first load if model hasn't been trained yet
    if not st.session_state.model_trained and not st.session_state.train_triggered:
        st.session_state.train_triggered = True
