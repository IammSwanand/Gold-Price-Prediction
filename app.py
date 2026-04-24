import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🥇 Gold Price Predictor", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for model training control
for _key, _default in [
    ('model_trained', False),
    ('train_triggered', False),
    ('cached_model', None),
    ('cached_data', None),
    ('cached_model_type', 'Linear Regression'),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Custom CSS
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

# ==================== DATA LOADING & PROCESSING ====================
@st.cache_data
def load_data(start_date, end_date):
    """Load GLD data from yfinance"""
    try:
        # Add 5 days buffer to end_date to ensure we get the most recent available data
        end_date_buffer = end_date + timedelta(days=5)
        data = yf.download('GLD', start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date_buffer.strftime('%Y-%m-%d'), auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        # Filter to end_date just to be safe (in case we got extra future dates)
        data = data[data.index <= end_date]
        return data[['Close']].dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_features(df, ma_short, ma_long, days_ahead=1):
    """Prepare features for modeling.
    Always computes both raw and normalised columns so any model can use them.
    """
    df_copy = df.copy()
    df_copy['S_short'] = df_copy['Close'].rolling(window=ma_short).mean()
    df_copy['S_long']  = df_copy['Close'].rolling(window=ma_long).mean()

    # Absolute target  (Linear Regression)
    df_copy['target_price'] = df_copy['Close'].shift(-days_ahead)

    # --- Normalised features (scale-invariant for tree models) ---
    # Ratios hover near 1.0 regardless of the price level → no extrapolation issue
    df_copy['S_short_norm'] = df_copy['S_short'] / df_copy['Close']
    df_copy['S_long_norm']  = df_copy['S_long']  / df_copy['Close']

    # Ratio target: next_price / current_price  (tree models predict this)
    df_copy['target_ratio'] = df_copy['Close'].shift(-days_ahead) / df_copy['Close']

    return df_copy.dropna()

def train_model(df, ma_short, ma_long, split_ratio, days_ahead=1, model_type='Linear Regression'):
    """Train the selected model.

    Linear Regression  → raw price features + absolute price target.
    XGBoost / RF       → normalised ratio features + ratio target
                         (scale-invariant, fixes the extrapolation problem).
    """
    df_prepared = prepare_features(df, ma_short, ma_long, days_ahead)
    split_idx   = int(len(df_prepared) * split_ratio)

    if model_type in ('XGBoost', 'Random Forest'):
        # Normalised features: always near 1.0 regardless of price level
        feature_cols = ['S_short_norm', 'S_long_norm']
        use_ratio    = True
        X = df_prepared[feature_cols]
        y = df_prepared['target_ratio']          # predict next/current ratio
    else:
        feature_cols = ['S_short', 'S_long']
        use_ratio    = False
        X = df_prepared[feature_cols]
        y = df_prepared['target_price']

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if model_type == 'XGBoost':
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            verbosity=0
        ).fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ).fit(X_train, y_train)
    else:
        model = LinearRegression().fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df_prepared, split_idx, feature_cols, use_ratio

def get_cointegration_stats(df, ma_short, ma_long, days_ahead=1):
    """Calculate cointegration statistics"""
    df_prepared = prepare_features(df, ma_short, ma_long, days_ahead)
    
    s_short = df_prepared['S_short'].dropna()
    s_long = df_prepared['S_long'].dropna()
    target_price = df_prepared['target_price'].dropna()
    
    # Ensure same length
    min_len = min(len(s_short), len(s_long), len(target_price))
    s_short, s_long, target_price = s_short[-min_len:], s_long[-min_len:], target_price[-min_len:]
    
    coint_short = coint(s_short, target_price)
    coint_long = coint(s_long, target_price)
    
    return {
        'short_pvalue': coint_short[1],
        'long_pvalue': coint_long[1],
        'short_stat': coint_short[0],
        'long_stat': coint_long[0]
    }

# ==================== SIDEBAR CONFIGURATION ====================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Training data period in expander
with st.sidebar.expander("📅 Training Data Period", expanded=False):
    date_preset = st.radio(
        "Select Period",
        options=["Last 5 Years", "Last 10 Years", "All Data (2012-Present)", "Custom Range"],
        help="Select the period for model training"
    )
    
    if date_preset == "Last 5 Years":
        train_start = datetime.now() - timedelta(days=5*365)
        train_end = datetime.now()
    elif date_preset == "Last 10 Years":
        train_start = datetime.now() - timedelta(days=10*365)
        train_end = datetime.now()
    elif date_preset == "All Data (2012-Present)":
        train_start = datetime(2012, 1, 1)
        train_end = datetime.now()
    else:  # Custom
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("Start Date", value=datetime(2012, 1, 1))
        with col2:
            train_end = st.date_input("End Date", value=datetime.now())

# Model parameters in expander
with st.sidebar.expander("⚙️ Model Parameters", expanded=False):
    ma_short = st.slider("Short Moving Average (days)", 1, 10, 3)
    ma_long = st.slider("Long Moving Average (days)", 5, 20, 9)
    train_split = st.slider("Train/Test Split", 0.5, 0.95, 0.8)

# Train Model button
if st.sidebar.button("🚀 Train Model", use_container_width=True, help="Click to train/retrain the model with current parameters"):
    st.session_state.train_triggered = True

# Initialize session state (kept for backward compatibility)
if 'train_triggered' not in st.session_state:
    st.session_state.train_triggered = False

# Auto-train on first load if model hasn't been trained yet
if not st.session_state.model_trained and not st.session_state.train_triggered:
    st.session_state.train_triggered = True

# ==================== MAIN DASHBOARD ====================
st.title("🥇 Gold Price Prediction Dashboard")
st.markdown("**Tracking: GLD - SPDR Gold Shares ETF**")
st.markdown(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Display metrics at top of page
st.markdown("---")

# Forecast Horizon + Model Selector side-by-side
_col_left, _col_right = st.columns(2)

with _col_left:
    days_ahead = st.selectbox(
        "🎯 Forecast Horizon",
        options=[1, 2],
        format_func=lambda x: f"{x} Day{'s' if x > 1 else ''} Ahead",
        help="Predict price for tomorrow (1 day) or day after tomorrow (2 days)",
        index=0
    )

with _col_right:
    selected_model = st.selectbox(
        "🤖 Prediction Model",
        options=["Linear Regression", "XGBoost", "Random Forest"],
        index=0,
        help="Linear Regression (default) · XGBoost (gradient boosting) · Random Forest (ensemble)"
    )

# If model type changed since last training, force retrain
if st.session_state.model_trained and selected_model != st.session_state.cached_model_type:
    st.info(f"⚠️ Model changed to **{selected_model}**. Click **🚀 Train Model** to apply.")

# Calculate forecast date based on the selected horizon
forecast_date = datetime.now() + timedelta(days=days_ahead)

st.markdown("---")

# Train model if triggered (by button click or auto-training on first load)
if st.session_state.train_triggered:
    is_auto_train = not st.session_state.model_trained
    spinner_msg = "⏳ Loading model with default parameters..." if is_auto_train else "🔄 Retraining model with updated parameters..."

    with st.spinner(spinner_msg):
        df = load_data(train_start, train_end)
        if df is None:
            st.stop()
        
        model, X_train, X_test, y_train, y_test, df_prepared, split_idx, feature_cols, use_ratio = train_model(
            df, ma_short, ma_long, train_split, days_ahead, selected_model
        )
        st.session_state.cached_model_type = selected_model
        coint_stats = get_cointegration_stats(df, ma_short, ma_long, days_ahead)

        # Cache the model in session state
        st.session_state.cached_model = model
        st.session_state.cached_data = {
            'df': df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'df_prepared': df_prepared,
            'split_idx': split_idx,
            'coint_stats': coint_stats,
            'feature_cols': feature_cols,
            'use_ratio': use_ratio,
        }
        st.session_state.model_trained = True
        st.session_state.train_triggered = False  # Reset trigger
    
    if not is_auto_train:
        st.success("✅ Model retrained successfully!")

# Use cached model and data
model = st.session_state.cached_model
cached_data = st.session_state.cached_data
df          = cached_data['df']
X_train     = cached_data['X_train']
X_test      = cached_data['X_test']
y_train     = cached_data['y_train']
y_test      = cached_data['y_test']
df_prepared = cached_data['df_prepared']
split_idx   = cached_data['split_idx']
coint_stats = cached_data['coint_stats']
feature_cols = cached_data.get('feature_cols', ['S_short', 'S_long'])
use_ratio    = cached_data.get('use_ratio', False)

# For tree models, compute price-equivalent predictions for R² / charts
_test_close  = df_prepared.iloc[split_idx:]['Close'].values
_test_prices = df_prepared.iloc[split_idx:]['target_price'].values

if use_ratio:
    _y_pred_raw    = model.predict(X_test)
    y_pred_prices  = _y_pred_raw * _test_close   # ratio → absolute price
    y_test_prices  = _test_prices
    _r2            = r2_score(y_test_prices, y_pred_prices)
else:
    y_pred_prices  = model.predict(X_test)
    y_test_prices  = y_test.values
    _r2            = model.score(X_test, y_test)

# Display metrics at top of page
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Model R² Score", f"{_r2*100:.2f}%")

with col2:
    st.metric("📈 Current Price", f"${df['Close'].iloc[-1]:.2f}")

with col3:
    price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100)
    st.metric("20-Day Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")

with col4:
    sharpe = (y_test.mean() - y_train.mean()) / (y_test.std() + 1e-10) * np.sqrt(252)
    st.metric("Sharpe Ratio (approx)", f"{sharpe:.2f}")

st.markdown("---")

# ==================== TABS FOR ORGANIZATION ====================
tab1, tab2, tab3 = st.tabs([
    "💡 Trading Signals", 
    "🔮 Price Predictions", 
    "📊 Price Analysis"
])

# ==================== TAB 1: CURRENT SIGNALS ====================
with tab1:
    st.subheader("💡 Current Trading Signal")
    st.markdown(f"*Forecast Date: {forecast_date.strftime('%Y-%m-%d')}*")
    
    # Get current data up to TODAY (not forecast date which is in the future)
    today_data = load_data(train_start, datetime.now())
    today_data_prep = prepare_features(today_data, ma_short, ma_long, days_ahead)
    
    # Extract features using the same columns the model was trained on
    current_features = today_data_prep[feature_cols].iloc[-1].values.reshape(1, -1)
    _raw_pred = model.predict(current_features)[0]
    if use_ratio:
        # Multiply ratio by current close to get absolute predicted price
        predicted_price = today_data['Close'].iloc[-1] * _raw_pred
    else:
        predicted_price = _raw_pred
    
    # Get today's actual closing price (the real current price, not shifted)
    current_price = today_data['Close'].iloc[-1]
    
    signal = "🟢 BUY" if current_price < predicted_price else "🔴 NO POSITION"
    signal_color = "green" if current_price < predicted_price else "red"
    
    col1, col2 = st.columns(2)
    
    day_label = f"{days_ahead} Days Ahead" if days_ahead > 1 else "Tomorrow"
    
    with col1:
        st.metric(f"Predicted {day_label} Closing Price", f"${predicted_price:.2f}")
    
    with col2:
        price_diff = predicted_price - current_price
        st.metric("Predicted Change", f"${price_diff:.2f}", delta=f"{(price_diff/current_price)*100:.2f}%")
    
    st.markdown("---")
    
    # Signal recommendation
    if current_price < predicted_price:
        st.success(f"### {signal}\nPrice is below predicted level. Consider BUYING.", icon="✅")
    else:
        st.warning(f"### {signal}\nPrice is at/above predicted level. HOLD position.", icon="⚠️")
    
    st.markdown("---")
    st.subheader("📊 Recent Data")
    
    recent_data = today_data_prep[['Close', 'S_short', 'S_long']].tail(10).copy()
    recent_data.columns = ['Price', f'{ma_short}D MA', f'{ma_long}D MA']
    st.dataframe(recent_data, width='stretch')

# ==================== TAB 2: PREDICTIONS ====================
with tab2:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Model Predictions vs Actual (Test Set)")
        
        # y_pred_prices / y_test_prices are always in absolute $ (pre-computed above)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_test.index,
            y=y_test_prices,
            name='Actual Price',
            line=dict(color='green', width=2),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=X_test.index,
            y=y_pred_prices,
            name='Predicted Price',
            line=dict(color='orange', width=2, dash='dot'),
            mode='lines'
        ))
        
        fig.update_layout(
            title="Predictions vs Actual",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=450,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        _mtype = st.session_state.cached_model_type

        if _mtype == 'Linear Regression':
            st.subheader("Model Coefficients")
            coef_data = {
                'Feature': [f'{ma_short}-Day MA', f'{ma_long}-Day MA', 'Intercept'],
                'Coefficient': [model.coef_[0], model.coef_[1], model.intercept_]
            }
            st.dataframe(coef_data, width='stretch')
            st.subheader("Model Equation")
            st.latex(
                f"Price = {model.coef_[0]:.4f} \\times S_{{{ma_short}}} + "
                f"{model.coef_[1]:.4f} \\times S_{{{ma_long}}} + {model.intercept_:.4f}"
            )

        elif _mtype in ('XGBoost', 'Random Forest'):
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                'Feature': [f'{ma_short}-Day MA', f'{ma_long}-Day MA'],
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'],
                y=fi_df['Feature'],
                orientation='h',
                marker_color=['#1f77b4', '#ff7f0e'][:len(fi_df)]
            ))
            fig_fi.update_layout(
                title=f"{_mtype} Feature Importance",
                xaxis_title="Importance Score",
                height=250,
                template='plotly_white',
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption(
                "📌 **XGBoost** uses gradient boosting on decision trees.\n\n"
                "📌 **Random Forest** averages many decision trees."
                if _mtype == 'XGBoost' else
                "📌 **Random Forest** averages predictions from many decision trees to reduce overfitting."
            )
    
    # Prediction errors (always in absolute $)
    st.subheader("📉 Prediction Errors")
    errors = y_test_prices - y_pred_prices
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"${np.mean(np.abs(errors)):.4f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"${np.sqrt(np.mean(errors**2)):.4f}")
    with col3:
        st.metric("Mean Error (Bias)", f"${np.mean(errors):.4f}")
    with col4:
        st.metric("Error Std Dev", f"${np.std(errors):.4f}")

# ==================== TAB 3: PRICE ANALYSIS ====================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Gold Price with Moving Averages")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Close Price',
            line=dict(color='gold', width=2)
        ))
        
        df_prepared_plot = df_prepared
        fig.add_trace(go.Scatter(
            x=df_prepared_plot.index,
            y=df_prepared_plot['S_short'],
            name=f'{ma_short}-Day MA',
            line=dict(color='blue', width=1.5, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_prepared_plot.index,
            y=df_prepared_plot['S_long'],
            name=f'{ma_long}-Day MA',
            line=dict(color='red', width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            title="Gold ETF Price Trends",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Price Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df['Close'],
            nbinsx=50,
            name='Price Distribution',
            marker_color='gold'
        ))
        fig_dist.update_layout(
            title="Gold Price Distribution",
            xaxis_title="Price ($)",
            yaxis_title="Frequency",
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_dist, width='stretch')
    
    # Price statistics
    st.subheader("📊 Price Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Min Price", f"${df['Close'].min():.2f}")
    with col2:
        st.metric("Max Price", f"${df['Close'].max():.2f}")
    with col3:
        st.metric("Avg Price", f"${df['Close'].mean():.2f}")
    with col4:
        st.metric("Std Dev", f"${df['Close'].std():.2f}")
    with col5:
        st.metric("Volatility", f"{(df['Close'].pct_change().std() * 100):.2f}%")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🥇 Gold Price Prediction Dashboard (GLD - SPDR Gold Shares) | Built with Streamlit</p>
        <p style='font-size: 0.8em; color: gray;'>Disclaimer: This is for educational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)
