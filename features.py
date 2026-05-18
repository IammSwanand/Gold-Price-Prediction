"""
Feature Engineering
Handles feature preparation and technical indicators
"""
import streamlit as st
import pandas as pd


@st.cache_data
def prepare_features(df, ma_short, ma_long, days_ahead=1):
    """
    Prepare features for modeling.
    Always computes both raw and normalised columns so any model can use them.
    """
    df_copy = df.copy()
    df_copy['S_short'] = df_copy['Close'].rolling(window=ma_short).mean()
    df_copy['S_long'] = df_copy['Close'].rolling(window=ma_long).mean()

    # Absolute target (Linear Regression)
    df_copy['target_price'] = df_copy['Close'].shift(-days_ahead)

    # --- Normalised features (scale-invariant for tree models) ---
    # Ratios hover near 1.0 regardless of the price level → no extrapolation issue
    df_copy['S_short_norm'] = df_copy['S_short'] / df_copy['Close']
    df_copy['S_long_norm'] = df_copy['S_long'] / df_copy['Close']

    # Ratio target: next_price / current_price (tree models predict this)
    df_copy['target_ratio'] = df_copy['Close'].shift(-days_ahead) / df_copy['Close']

    return df_copy.dropna()
