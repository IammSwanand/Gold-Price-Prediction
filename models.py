"""
Gold Price Prediction Models
Handles model training and selection
"""
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from features import prepare_features


@st.cache_data
def load_data(start_date, end_date):
    """Load GLD data from yfinance with buffer for latest data"""
    try:
        # Add 5 days buffer to end_date to ensure we get the most recent available data
        end_date_buffer = end_date + timedelta(days=5)
        data = yf.download(
            'GLD',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date_buffer.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        # Filter to end_date just to be safe (in case we got extra future dates)
        data = data[data.index <= end_date]
        return data[['Close']].dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def train_model(df, ma_short, ma_long, split_ratio, days_ahead=1, model_type='Linear Regression'):
    """
    Train the selected model.

    Linear Regression  → raw price features + absolute price target.
    XGBoost / RF       → normalised ratio features + ratio target
                         (scale-invariant, fixes the extrapolation problem).
    """
    df_prepared = prepare_features(df, ma_short, ma_long, days_ahead)
    split_idx = int(len(df_prepared) * split_ratio)

    if model_type in ('XGBoost', 'Random Forest'):
        # Normalised features: always near 1.0 regardless of price level
        feature_cols = ['S_short_norm', 'S_long_norm']
        use_ratio = True
        X = df_prepared[feature_cols]
        y = df_prepared['target_ratio']  # predict next/current ratio
    else:
        feature_cols = ['S_short', 'S_long']
        use_ratio = False
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
        ).fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
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
