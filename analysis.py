"""
Statistical Analysis
Handles cointegration and other statistical metrics
"""
from statsmodels.tsa.stattools import coint
from features import prepare_features


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
