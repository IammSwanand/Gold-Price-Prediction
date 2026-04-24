# 🥇 Gold Price Prediction Dashboard

An interactive machine learning dashboard for predicting GLD (SPDR Gold Shares ETF) prices using multiple ML models, built with Python and Streamlit.

---

## 📌 Project Description

This project builds a **data-driven trading signal system** for Gold ETF (GLD) price prediction. It uses historical daily closing prices fetched live from Yahoo Finance and applies machine learning models to forecast the next day's (or next 2 days') closing price.

The core idea is rooted in **time series feature engineering** — short-term and long-term moving averages of the GLD price are used as input features to capture recent momentum and medium-term trend. A cointegration test (Engle-Granger) statistically validates that these features are meaningfully related to the next-day price, justifying their use as predictors.

The dashboard generates a **BUY / NO POSITION** trading signal:
- 🟢 **BUY** — If today's price is below the model's predicted next-day price (upward movement expected)
- 🔴 **NO POSITION** — If today's price is at or above the prediction (no edge, stay out)

---

## 📊 Analysis

### Data
- **Asset**: GLD — SPDR Gold Shares ETF (proxy for physical gold)
- **Source**: Yahoo Finance via `yfinance`
- **Range**: 2012–Present (~3,500+ trading days)
- **Features**: 3-day Moving Average (S₃), 9-day Moving Average (S₉)
- **Target**: Next-day (or 2-day) closing price

### Statistical Validation
Before training, an **Engle-Granger Cointegration Test** is applied to confirm that both moving averages are statistically cointegrated with the next-day price:
- S₃ p-value: `~3.13e-16` → strongly cointegrated ✅
- S₉ p-value: `~1.27e-15` → strongly cointegrated ✅

This confirms that the features are not spurious and carry genuine predictive signal.

### Models Supported

| Model | Type | Features Used | Target |
|-------|------|--------------|--------|
| **Linear Regression** | Classical ML | Raw MA values ($) | Absolute next-day price ($) |
| **XGBoost** | Gradient Boosting | Normalised MA ratios | Price ratio (next/current) |
| **Random Forest** | Ensemble | Normalised MA ratios | Price ratio (next/current) |

### Model Comparison Methodology

Tree-based models (XGBoost, Random Forest) **cannot extrapolate** beyond the price range seen during training — a critical issue since GLD has risen from ~$120 (2012) to ~$430+ (2025). To fix this:

- **Features are normalised**: `S₃ / Close` and `S₉ / Close` instead of raw $ values. These ratios are always ~1.0 regardless of price level, making the model scale-invariant.
- **Target is a ratio**: `next_day_price / today_price` instead of absolute price. The predicted ratio is then multiplied by the current price to get the final prediction.
- **Linear Regression** continues to use raw price features since it can extrapolate naturally.

A full **side-by-side comparison** of all three models is performed in the research notebook (`Gold Price Prediction (2).ipynb`) across four metrics:

| Metric | Meaning |
|--------|---------|
| **R² (%)** | % of price variance explained (higher = better) |
| **MAE ($)** | Mean absolute prediction error in dollars |
| **RMSE ($)** | Root mean squared error (penalises large errors) |
| **MAPE (%)** | Mean absolute percentage error |

### Results (Linear Regression, All Data 2012–Present)
| Metric | Value |
|--------|-------|
| R² Score | ~99.7% |
| Sharpe Ratio | ~1.82 |
| Signal Logic | Price < Predicted → BUY |

> ⚠️ The high R² is partly due to the near-unit-root nature of price series (prices are strongly autocorrelated). Always validate with out-of-sample data.

---

## ✅ Conclusion

This project demonstrates that **moving average-based features are strong short-term predictors** of gold ETF prices, with both S₃ and S₉ showing statistically significant cointegration with next-day prices. The linear model achieves a very high R² score and a Sharpe Ratio above 1.5, indicating the strategy would have generated meaningful risk-adjusted returns historically.

**Model comparison findings:**
- **Linear Regression** serves as a strong, interpretable baseline with near-perfect R² on price data
- **XGBoost** captures non-linear relationships between MA ratios and price movements; better generalises to unseen price levels using normalised features
- **Random Forest** provides similar scale-invariant predictions via ensemble averaging, reducing variance at the cost of some bias
- All three models are compared head-to-head in the notebook using R², MAE, RMSE, and MAPE on the same 20% test set

Key takeaways:
- Normalising features (ratio-based) is **essential** for tree models to work correctly on financial price data that trends over time
- The signal system is simple but statistically grounded via cointegration testing
- The dashboard makes all three models fully interactive and configurable for further experimentation
- Future improvements: adding more features (RSI, MACD, volume), implementing LSTM for sequence modeling, and adding proper backtesting with transaction costs

---

## 🗂️ Project Structure

```
Gold Price Prediction/
│
├── app.py                          # Main Streamlit dashboard
│                                   #   ↳ Linear Regression, XGBoost, Random Forest
│                                   #   ↳ Model selector dropdown + auto-training
├── Gold Price Prediction (2).ipynb # Research & prototyping notebook
│                                   #   ↳ EDA, cointegration testing
│                                   #   ↳ Multi-model training & comparison charts
├── README.md                       # Project documentation
└── .venv/                          # Python virtual environment
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `yfinance` | Live market data (GLD ETF) |
| `scikit-learn` | Linear Regression, Random Forest |
| `xgboost` | XGBoost Regressor |
| `statsmodels` | Cointegration testing |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data processing |

---

## 🚀 Getting Started

### 1. Clone / Download the project
```bash
cd "Gold Price Prediction"
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install streamlit yfinance scikit-learn xgboost statsmodels plotly pandas numpy
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Dashboard Features

| Feature | Description |
|---------|-------------|
| **Auto-training** | Model trains automatically on first load (Linear Regression + default params) |
| **Model Selector** | Switch between Linear Regression, XGBoost, Random Forest |
| **Forecast Horizon** | Predict 1 day or 2 days ahead |
| **Trading Signal** | Live BUY / NO POSITION signal based on today's data |
| **Configurable Params** | Adjust MA windows, train/test split, date range |
| **Model Insights** | Coefficients (LR) or Feature Importance chart (XGBoost/RF) |
| **Error Metrics** | MAE, RMSE, Bias, Std Dev on test set |
| **Price Analysis** | Historical price chart + distribution + statistics |

---

## ⚠️ Disclaimer

This dashboard is built for **educational and research purposes only**. It does not constitute financial advice. Past model performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 👤 Author

Gold Price Prediction Dashboard — built with Python & Streamlit.
