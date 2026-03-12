import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import h5py
import json
import os
import tempfile
import shutil
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

# Portfolio Persistent Storage
PORTFOLIO_FILE = "portfolio_data.json"

def save_portfolio_to_file():
    """Save portfolio data to JSON file for persistence across sessions"""
    portfolio_data = {
        'portfolio': st.session_state.portfolio,
        'created_at': st.session_state.portfolio_created_at,
        'history': st.session_state.portfolio_history
    }
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio_data, f, indent=4)
    except Exception as e:
        st.warning(f"Could not save portfolio: {e}")

def load_portfolio_from_file():
    """Load portfolio data from JSON file if it exists"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio_data = json.load(f)
            return portfolio_data
        except Exception as e:
            st.warning(f"Could not load portfolio: {e}")
            return None
    return None

# Load persistent portfolio data
persistent_portfolio = load_portfolio_from_file()

# -------------------------------
# Load LSTM Model with H5 Compatibility Fix
# -------------------------------
def load_model_with_batch_shape_fix(model_path):
    """Fix batch_shape and dtype_policy incompatibility in h5 models"""
    try:
        # Try with compile=False first (bypasses dtype serialization issues)
        return load_model(model_path, compile=False)
    except (TypeError, ValueError) as e:
        if "batch_shape" in str(e) or "DTypePolicy" in str(e):
            # Create a temporary copy of the model
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                shutil.copy(model_path, tmp_path)
                
                # Modify the h5 file to fix compatibility issues
                with h5py.File(tmp_path, 'r+') as f:
                    if 'model_config' in f.attrs:
                        config_str = f.attrs['model_config']
                        if isinstance(config_str, bytes):
                            config_str = config_str.decode('utf-8')
                        
                        config = json.loads(config_str)
                        
                        # Remove dtype_policy from the entire config to avoid serialization issues
                        def clean_config(obj):
                            if isinstance(obj, dict):
                                # Remove dtype_policy and related fields
                                obj.pop('dtype_policy', None)
                                obj.pop('dtype', None)
                                # Recursively clean nested dicts
                                for key in list(obj.keys()):
                                    if isinstance(obj[key], (dict, list)):
                                        clean_config(obj[key])
                            elif isinstance(obj, list):
                                for item in obj:
                                    if isinstance(item, (dict, list)):
                                        clean_config(item)
                        
                        clean_config(config)
                        
                        # Fix batch_shape to input_shape in InputLayers
                        if 'config' in config and 'layers' in config['config']:
                            for layer in config['config']['layers']:
                                if layer.get('class_name') == 'InputLayer':
                                    layer_config = layer.get('config', {})
                                    if 'batch_shape' in layer_config:
                                        batch_shape = layer_config.pop('batch_shape')
                                        if batch_shape and len(batch_shape) > 1:
                                            layer_config['input_shape'] = batch_shape[1:]
                        
                        # Write back the cleaned config
                        f.attrs['model_config'] = json.dumps(config)
                
                # Load from the modified temporary file
                model = load_model(tmp_path, compile=False)
                return model
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        raise

model = load_model_with_batch_shape_fix("lstm_model.h5")

# Portfolio Simulator Functions
# ==============================
def get_stock_price(ticker):
    """Get current stock price"""
    try:
        data = yf.download(ticker, period='5d', progress=False)
        if data is not None and not data.empty:
            price = float(data['Close'].iloc[-1])
            return price if price > 0 else None
        
        # Fallback
        tick = yf.Ticker(ticker)
        price = tick.info.get('currentPrice')
        return float(price) if price and price > 0 else None
    except:
        return None


def get_stock_volatility(ticker, period=252):
    """Calculate annualized volatility"""
    try:
        data = yf.download(ticker, period='1y', progress=False)
        if data is not None and not data.empty and len(data) > 1:
            returns = data['Close'].pct_change()
            volatility = float(returns.std() * np.sqrt(period))
            return volatility if volatility > 0 else None
        return None
    except:
        return None


def get_stock_beta(ticker):
    """Get stock beta value"""
    try:
        tick = yf.Ticker(ticker)
        beta = tick.info.get('beta')
        if beta is not None:
            return float(beta)
        return 1.0
    except:
        return 1.0


def calculate_portfolio_metrics(portfolio_dict):
    """Calculate portfolio metrics (value, P&L, allocation, risk)"""
    if not portfolio_dict or len(portfolio_dict) == 0:
        return None
    
    metrics = {
        'tickers': [],
        'investments': [],
        'current_prices': [],
        'current_values': [],
        'shares': [],
        'p_l': [],
        'p_l_percent': [],
        'allocation': [],
        'volatility': [],
        'beta': [],
    }
    
    total_invested = 0
    total_current = 0
    
    # Get data for each stock
    for ticker, amount in portfolio_dict.items():
        price = get_stock_price(ticker.upper())
        
        if price is None:
            st.warning(f"Could not fetch price for {ticker}")
            continue
        
        shares = amount / price
        current_value = shares * price
        p_l = current_value - amount
        p_l_percent = (p_l / amount * 100) if amount > 0 else 0
        volatility = get_stock_volatility(ticker.upper())
        beta = get_stock_beta(ticker.upper())
        
        # Handle volatility safely
        if volatility is None:
            vol_value = 0
        else:
            try:
                vol_value = float(volatility)
            except:
                vol_value = 0
        
        # Handle beta safely
        if beta is None:
            beta_value = 1.0
        else:
            try:
                beta_value = float(beta)
            except:
                beta_value = 1.0
        
        metrics['tickers'].append(ticker.upper())
        metrics['investments'].append(amount)
        metrics['current_prices'].append(price)
        metrics['current_values'].append(current_value)
        metrics['shares'].append(shares)
        metrics['p_l'].append(p_l)
        metrics['p_l_percent'].append(p_l_percent)
        metrics['volatility'].append(vol_value)
        metrics['beta'].append(beta_value)
        
        total_invested += amount
        total_current += current_value
    
    # Calculate portfolio-level metrics
    if len(metrics['tickers']) > 0:
        metrics['total_invested'] = total_invested
        metrics['total_current'] = total_current
        metrics['total_p_l'] = total_current - total_invested
        metrics['total_p_l_percent'] = (metrics['total_p_l'] / total_invested * 100) if total_invested > 0 else 0
        
        # Asset allocation percentages
        metrics['allocation'] = [v / total_current * 100 for v in metrics['current_values']] if total_current > 0 else []
        
        # Portfolio-weighted volatility and beta
        weights = [v / total_current for v in metrics['current_values']] if total_current > 0 else []
        metrics['portfolio_volatility'] = sum(v * w for v, w in zip(metrics['volatility'], weights))
        metrics['portfolio_beta'] = sum(b * w for b, w in zip(metrics['beta'], weights))
        
        return metrics
    
    return None

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Stock Market Analysis",
    layout="wide"
)

st.title("Stock Market Analysis Dashboard")
st.markdown("Analyze stocks using **Moving Averages, RSI, Strategy Backtesting & LSTM Prediction**")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Stock Settings")

ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    help="Examples: AAPL, TSLA, INFY.NS, TCS.NS"
)

ticker = ticker_input.upper().strip()

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ticker

if ticker != st.session_state.last_ticker:
    st.cache_data.clear()
    st.session_state.last_ticker = ticker
    st.rerun()

period = st.sidebar.selectbox(
    "Select Time Period",
    ["6mo", "1y", "2y", "5y"]
)

sma_short = st.sidebar.slider("Short SMA Window", 5, 50, 20)
sma_long = st.sidebar.slider("Long SMA Window", 20, 200, 50) 
rsi_period = st.sidebar.slider("RSI Period", 7, 30, 14)

# Refresh button
if st.sidebar.button("🔄 Refresh Data & Clear Cache", help="Force reload"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Portfolio Simulator
# --------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📈 Portfolio Simulator")

# Initialize session state for portfolio storage
if 'portfolio' not in st.session_state:
    if persistent_portfolio:
        st.session_state.portfolio = persistent_portfolio['portfolio']
        st.session_state.portfolio_created_at = persistent_portfolio['created_at']
        st.session_state.portfolio_history = persistent_portfolio['history']
    else:
        st.session_state.portfolio = {}
        st.session_state.portfolio_created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.portfolio_history = []

# Portfolio builder in expander
with st.sidebar.expander("🔨 Build Portfolio", expanded=True):
    st.subheader("Add Holdings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        new_ticker = st.text_input(
            "Stock Ticker",
            placeholder="AAPL",
            key="new_ticker"
        )
    
    with col2:
        new_amount = st.number_input(
            "Investment ($)",
            min_value=100.0,
            step=100.0,
            value=1000.0,
            key="new_amount"
        )
    
    if st.button("➕ Add to Portfolio", use_container_width=True):
        if new_ticker and new_amount > 0:
            st.session_state.portfolio[new_ticker.upper()] = new_amount
            # Add to history
            st.session_state.portfolio_history.append({
                'action': 'ADD',
                'ticker': new_ticker.upper(),
                'amount': new_amount,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_portfolio_to_file()
            st.success(f"Added ${new_amount} of {new_ticker.upper()}")
        else:
            st.error("Enter valid ticker and amount")
    
    # Portfolio Management Controls
    col_clear, col_reset = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state.portfolio = {}
            st.session_state.portfolio_created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.portfolio_history = []
            save_portfolio_to_file()
            st.info("Portfolio cleared")
            st.rerun()
    
    # Session State Info
    if st.session_state.portfolio:
        st.markdown("---")
        st.caption(f"📅 Created: {st.session_state.portfolio_created_at}")
        st.caption(f"📊 Holdings: {len(st.session_state.portfolio)}")
        st.caption(f"📝 Changes: {len(st.session_state.portfolio_history)}")
    
    if st.session_state.portfolio:
        st.subheader("Current Holdings")
        for ticker, amount in st.session_state.portfolio.items():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(f"**{ticker}**")
            with col_b:
                if st.button("❌", key=f"remove_{ticker}", use_container_width=True):
                    del st.session_state.portfolio[ticker]
                    save_portfolio_to_file()
                    st.rerun()
            st.caption(f"${amount:.2f}")

# Portfolio View in expander
with st.sidebar.expander("📊 View Portfolio", expanded=False):
    if st.session_state.portfolio:
        # Session State Information
        st.caption(f"**Session Storage Status:**")
        st.caption(f"✅ portfolio - {len(st.session_state.portfolio)} holdings")
        st.caption(f"📅 Created: {st.session_state.portfolio_created_at}")
        st.markdown("---")
        
        st.subheader("Portfolio Metrics")
        
        # Calculate metrics
        with st.spinner("Calculating..."):
            metrics = calculate_portfolio_metrics(st.session_state.portfolio)
        
        if metrics:
            # Portfolio Summary Metrics
            st.metric("Total Value", f"${metrics['total_current']:.2f}", f"${metrics['total_p_l']:.2f}")
            st.metric("Return %", f"{metrics['total_p_l_percent']:.2f}%")
            st.metric("Portfolio Vol.", f"{metrics['portfolio_volatility']*100:.2f}%")
            st.metric("Portfolio Beta", f"{metrics['portfolio_beta']:.2f}")
            
            st.markdown("---")
            
            # Holdings Table
            st.subheader("Holdings")
            holdings_data = {
                'Stock': metrics['tickers'],
                'Qty': [f"{s:.2f}" for s in metrics['shares']],
                'Value': [f"${v:.0f}" for v in metrics['current_values']],
                'Return': [f"{p:.1f}%" for p in metrics['p_l_percent']],
                '%': [f"{a:.0f}%" for a in metrics['allocation']],
            }
            st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)
        else:
            st.info("Could not calculate metrics.")
    else:
        st.info("📭 No portfolio created yet. Add stocks in the 'Build Portfolio' section.")

# -------------------------------
# Fetch Stock Data
# -------------------------------
def load_data(ticker, period):
    """Load stock data safely - NO CACHING for fresh data"""
    try:
        df = yf.download(
            ticker.upper(),
            period=period,
            progress=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        if "Close" not in df.columns:
            return None

        df = df.dropna(subset=["Close"])

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Fetch Fundamental Data
# -------------------------------
def get_fundamentals(ticker):
    """Fetch fundamental metrics for the stock"""
    try:
        tick = yf.Ticker(ticker)
        info = tick.info
        
        fundamentals = {
            'Market Cap': info.get('marketCap'),
            'P/E Ratio': info.get('trailingPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Price to Book': info.get('priceToBook'),
            'Dividend Yield': info.get('dividendYield'),
            '52W High': info.get('fiftyTwoWeekHigh'),
            '52W Low': info.get('fiftyTwoWeekLow'),
            'Beta': info.get('beta'),
            'Employees': info.get('fullTimeEmployees'),
            'Website': info.get('website'),
            'Industry': info.get('industry'),
            'Sector': info.get('sector'),
        }
        return fundamentals
    except Exception as e:
        st.warning(f"Could not fetch fundamentals: {str(e)}")
        return None


# Get Dividend History
# -------------------------------
def get_dividend_history(ticker):
    """Fetch dividend history for the stock"""
    try:
        tick = yf.Ticker(ticker)
        dividends = tick.dividends
        
        if dividends is not None and not dividends.empty:
            df_div = pd.DataFrame(dividends)
            df_div.columns = ['Dividend']
            df_div.index.name = 'Date'
            df_div = df_div.reset_index()
            return df_div.sort_values('Date', ascending=False)
        return None
    except Exception as e:
        st.warning(f"Could not fetch dividend history: {str(e)}")
        return None


# Get Earnings Calendar
# -------------------------------
def get_earnings_info(ticker):
    """Fetch earnings information"""
    try:
        tick = yf.Ticker(ticker)
        info = tick.info
        
        earnings_data = {
            'Earnings Date': info.get('earningsDate'),
            'Earnings Average': info.get('epsTrailingTwelveMonths'),
            'Earnings Growth': info.get('earningsGrowth'),
            'Revenue': info.get('totalRevenue'),
            'Profit Margin': info.get('profitMargins'),
        }
        return earnings_data
    except Exception as e:
        st.warning(f"Could not fetch earnings info: {str(e)}")
        return None







# Load data (ticker and period are now properly passed)
df = load_data(ticker, period)

if df is None or df.empty or len(df) < 2:
    st.warning(f"⚠️ Could not fetch live data for {ticker}.")
    
    # Offer demo mode
    if st.button("📊 Load Demo Data (Sample AAPL)"):
        st.info("Loading demo data...")
        # Generate demo data
        dates = pd.date_range(start='2023-01-01', periods=250)
        np.random.seed(42)
        closes = 150 + np.cumsum(np.random.randn(250) * 2)
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': closes,
            'Open': closes - np.abs(np.random.randn(250)),
            'High': closes + np.abs(np.random.randn(250)),
            'Low': closes - np.abs(np.random.randn(250)),
            'Volume': np.random.randint(1000000, 5000000, 250)
        })
        st.success("✓ Demo data loaded")
    else:
        st.error(f"❌ Could not fetch data for {ticker}. Please check your internet connection or:")
        st.info("""
        **Troubleshooting:**
        - Check ticker symbol (AAPL, MSFT, AMZN, GOOGL for US stocks)
        - Use .NS extension for Indian stocks (INFY.NS, TCS.NS)
        - Check internet connection
        - Try uploading a CSV file instead
        """)
        
        # CSV upload option
        st.subheader("📁 Or Upload Your Own Data")
        uploaded_file = st.file_uploader("Upload CSV with columns: Date, Close, Open, High, Low, Volume", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['Date'])
                st.success("✓ CSV loaded successfully")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                st.stop()
        else:
            st.stop()

if df is None or df.empty or len(df) < 2:
    st.stop()

st.success(f"✓ Loaded {len(df)} records for {ticker}")

# Display loaded data info
st.sidebar.write(f"✓ Loaded {len(df)} records")
st.sidebar.write(f"Date range: {df['Date'].min() if 'Date' in df.columns else df.index.min()} to {df['Date'].max() if 'Date' in df.columns else df.index.max()}")

# -------------------------------
# Indicator Calculations
# -------------------------------
df['SMA_Short'] = df['Close'].rolling(sma_short).mean()
df['SMA_Long'] = df['Close'].rolling(sma_long).mean()

# RSI
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(rsi_period).mean()
avg_loss = loss.rolling(rsi_period).mean()

rs = avg_gain / avg_loss

df['RSI'] = 100 - (100/(1+rs))

# -------------------------------
# Trading Signals
# -------------------------------
df['Signal'] = 0
df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1

# -------------------------------
# Backtesting
# -------------------------------
df['Returns'] = df['Close'].pct_change().fillna(0)

df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)

cumulative_market = (1 + df['Returns']).cumprod()
cumulative_strategy = (1 + df['Strategy_Returns']).cumprod()

# -------------------------------
# LSTM Prediction Preparation
# -------------------------------
lookback = 60

scaler = MinMaxScaler(feature_range=(0,1))

data = df[['Close']].values

scaled_data = scaler.fit_transform(data)

X = []

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i,0])

X = np.array(X)

X = np.reshape(X,(X.shape[0],X.shape[1],1))

# -------------------------------
# LSTM Predictions
# -------------------------------
predictions = model.predict(X)

predictions = scaler.inverse_transform(predictions)

df_lstm = df.iloc[lookback:].copy()

df_lstm['LSTM_Prediction'] = predictions

# -------------------------------
# Stock Data Preview
# -------------------------------
st.subheader(f"Stock Data Preview ({ticker})")
st.dataframe(df.tail(10))

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Fundamentals", "💰 Earnings", "📈 Dividends"])

# TAB 1: Fundamentals
# -------------------------------
with tab1:
    st.subheader("Company Fundamentals")
    
    fundamentals = get_fundamentals(ticker)
    
    if fundamentals:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if fundamentals.get('Market Cap'):
                market_cap = fundamentals['Market Cap']
                if isinstance(market_cap, (int, float)):
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", str(market_cap))
            
            if fundamentals.get('P/E Ratio'):
                st.metric("P/E Ratio", f"{fundamentals['P/E Ratio']:.2f}" if isinstance(fundamentals['P/E Ratio'], (int, float)) else "N/A")
            
            if fundamentals.get('Dividend Yield'):
                div_yield = fundamentals['Dividend Yield']
                if isinstance(div_yield, (int, float)):
                    st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
                else:
                    st.metric("Dividend Yield", "N/A")
        
        with col2:
            if fundamentals.get('PEG Ratio'):
                st.metric("PEG Ratio", f"{fundamentals['PEG Ratio']:.2f}" if isinstance(fundamentals['PEG Ratio'], (int, float)) else "N/A")
            
            if fundamentals.get('Price to Book'):
                st.metric("Price to Book", f"{fundamentals['Price to Book']:.2f}" if isinstance(fundamentals['Price to Book'], (int, float)) else "N/A")
            
            if fundamentals.get('Beta'):
                st.metric("Beta", f"{fundamentals['Beta']:.2f}" if isinstance(fundamentals['Beta'], (int, float)) else "N/A")
        
        with col3:
            if fundamentals.get('52W High'):
                st.metric("52W High", f"${fundamentals['52W High']:.2f}" if isinstance(fundamentals['52W High'], (int, float)) else "N/A")
            
            if fundamentals.get('52W Low'):
                st.metric("52W Low", f"${fundamentals['52W Low']:.2f}" if isinstance(fundamentals['52W Low'], (int, float)) else "N/A")
        
        # Company Info
        st.subheader("Company Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"**Sector:** {fundamentals.get('Sector', 'N/A')}")
            st.write(f"**Industry:** {fundamentals.get('Industry', 'N/A')}")
            st.write(f"**Website:** {fundamentals.get('Website', 'N/A')}")
        
        with info_col2:
            if fundamentals.get('Employees'):
                employees = fundamentals['Employees']
                if isinstance(employees, (int, float)):
                    st.write(f"**Employees:** {employees:,.0f}")
                else:
                    st.write(f"**Employees:** {employees}")
    else:
        st.info("Fundamental data not available for this ticker.")

# TAB 2: Earnings Calendar
# -------------------------------
with tab2:
    st.subheader("Earnings Information")
    
    earnings_info = get_earnings_info(ticker)
    
    if earnings_info:
        col1, col2 = st.columns(2)
        
        with col1:
            if earnings_info.get('Earnings Date'):
                st.write(f"**Earnings Date:** {earnings_info.get('Earnings Date')}")
            
            if earnings_info.get('Earnings Average'):
                st.metric("EPS (TTM)", f"${earnings_info.get('Earnings Average'):.2f}" if isinstance(earnings_info.get('Earnings Average'), (int, float)) else "N/A")
            
            if earnings_info.get('Revenue'):
                revenue = earnings_info.get('Revenue')
                if isinstance(revenue, (int, float)):
                    st.metric("Total Revenue", f"${revenue/1e9:.2f}B")
        
        with col2:
            if earnings_info.get('Earnings Growth'):
                growth = earnings_info.get('Earnings Growth')
                if isinstance(growth, (int, float)):
                    st.metric("Earnings Growth", f"{growth*100:.2f}%")
            
            if earnings_info.get('Profit Margin'):
                margin = earnings_info.get('Profit Margin')
                if isinstance(margin, (int, float)):
                    st.metric("Profit Margin", f"{margin*100:.2f}%")
    else:
        st.info("Earnings data not available for this ticker.")

# TAB 3: Dividend History
# -------------------------------
with tab3:
    st.subheader("Dividend History")
    
    dividends = get_dividend_history(ticker)
    
    if dividends is not None and not dividends.empty:
        st.dataframe(dividends, use_container_width=True)
        
        # Plot dividend trend
        fig_div, ax_div = plt.subplots(figsize=(12, 4))
        ax_div.bar(dividends['Date'], dividends['Dividend'], color='green', alpha=0.7)
        ax_div.set_title("Dividend Payment History")
        ax_div.set_xlabel("Date")
        ax_div.set_ylabel("Dividend Amount ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig_div)
        
        total_div = dividends['Dividend'].sum()
        st.metric("Total Dividends (shown period)", f"${total_div:.2f}")
    else:
        st.info("No dividend data available for this ticker.")



# ======================================
# Portfolio Visualization
# ======================================
if st.session_state.portfolio and len(st.session_state.portfolio) > 0:
    st.markdown("---")
    st.subheader("💼 Portfolio Analysis Dashboard")
    
    with st.spinner("Loading portfolio data..."):
        portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio)
    
    if portfolio_metrics:
        # Summary Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Invested",
                f"${portfolio_metrics['total_invested']:,.2f}"
            )
        
        with col2:
            st.metric(
                "Current Value",
                f"${portfolio_metrics['total_current']:,.2f}",
                f"${portfolio_metrics['total_p_l']:,.2f}"
            )
        
        with col3:
            p_l_color = "green" if portfolio_metrics['total_p_l_percent'] >= 0 else "red"
            st.metric(
                "Return",
                f"{portfolio_metrics['total_p_l_percent']:.2f}%"
            )
        
        with col4:
            st.metric(
                "Portfolio Volatility",
                f"{portfolio_metrics['portfolio_volatility']*100:.2f}%"
            )
        
        with col5:
            st.metric(
                "Portfolio Beta",
                f"{portfolio_metrics['portfolio_beta']:.2f}"
            )
        
        # Visualizations Row
        col_viz1, col_viz2 = st.columns([1, 1])
        
        # Asset Allocation Pie Chart
        with col_viz1:
            fig_alloc, ax_alloc = plt.subplots(figsize=(6, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(portfolio_metrics['tickers'])))
            wedges, texts, autotexts = ax_alloc.pie(
                portfolio_metrics['allocation'],
                labels=portfolio_metrics['tickers'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax_alloc.set_title("Asset Allocation", fontsize=14, fontweight='bold')
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
            st.pyplot(fig_alloc)
        
        # Risk Distribution (Volatility by Stock)
        with col_viz2:
            fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
            bars = ax_risk.bar(portfolio_metrics['tickers'], 
                               [v*100 for v in portfolio_metrics['volatility']],
                               color=colors)
            ax_risk.set_ylabel('Volatility (%)', fontsize=11)
            ax_risk.set_title("Risk Distribution (Volatility)", fontsize=14, fontweight='bold')
            ax_risk.grid(axis='y', alpha=0.3)
            st.pyplot(fig_risk)
        
        st.markdown("---")

# ======================================
# AI Market Story Generator
# ======================================

def analyze_volume_trend(df_data):
    """Analyze volume trend (Increasing, Decreasing, or Stable)"""
    if len(df_data) < 20:
        return "Stable", 0
    
    recent_volume = df_data['Volume'].tail(10).mean()
    past_volume = df_data['Volume'].tail(20).head(10).mean()
    
    volume_change = (recent_volume - past_volume) / past_volume * 100 if past_volume > 0 else 0
    
    if volume_change > 10:
        trend = "Increasing"
    elif volume_change < -10:
        trend = "Decreasing"
    else:
        trend = "Stable"
    
    confidence = min(abs(volume_change) / 30 * 100, 100)
    return trend, int(confidence)

def get_price_momentum(df_data):
    """Calculate price momentum (acceleration of price change)"""
    if len(df_data) < 5:
        return "Neutral", 0
    
    recent_5 = df_data['Close'].tail(5).pct_change().mean() * 100
    
    if recent_5 > 1:
        momentum = "Strong Upward"
        strength = min(abs(recent_5) / 5 * 100, 100)
    elif recent_5 < -1:
        momentum = "Strong Downward"
        strength = min(abs(recent_5) / 5 * 100, 100)
    else:
        momentum = "Neutral"
        strength = 50
    
    return momentum, int(strength)

def generate_market_story(df_data, ticker, trend, confidence, sma_short_val, sma_long_val):
    """Generate a human-like narrative explanation of market behavior"""
    
    if len(df_data) < 20:
        return f"{ticker} lacks sufficient data for analysis."
    
    # Get analysis data
    volume_trend, volume_confidence = analyze_volume_trend(df_data)
    momentum, momentum_strength = get_price_momentum(df_data)
    
    latest = df_data.iloc[-1]
    current_price = latest['Close']
    sma_s = latest[f'SMA_Short'] if f'SMA_Short' in latest else None
    sma_l = latest[f'SMA_Long'] if f'SMA_Long' in latest else None
    rsi_val = latest['RSI'] if 'RSI' in latest else 50
    
    # Price position relative to SMAs
    price_vs_sma_s = "above" if sma_s and current_price > sma_s else "below"
    price_vs_sma_l = "above" if sma_l and current_price > sma_l else "below"
    
    # Volume strength indicator
    volume_strength = "strong" if volume_confidence > 70 else "moderate" if volume_confidence > 40 else "weak"
    
    # RSI interpretation
    if rsi_val > 70:
        rsi_interpretation = "overbought conditions"
    elif rsi_val < 30:
        rsi_interpretation = "oversold conditions"
    else:
        rsi_interpretation = "neutral momentum"
    
    # Build the story
    stories = []
    
    if trend == "Bullish":
        if volume_trend == "Increasing":
            stories.append(
                f"{ticker} stock is currently showing **bullish momentum** due to {volume_strength} "
                f"increasing volume and positive trend strength. The price is trading {price_vs_sma_s} "
                f"its short-term moving average and {price_vs_sma_l} its long-term average, indicating "
                f"sustained upward pressure. With {rsi_interpretation}, the bullish setup appears robust."
            )
            
            stories.append(
                f"Strong buying interest is evident in {ticker}, as rising volume accompanies the "
                f"uptrend. The stock is maintaining positions {price_vs_sma_s} the short-term average "
                f"and is clearly in an uptrend above the long-term average. This {volume_strength}-volume "
                f"rally suggests conviction from market participants."
            )
        else:
            stories.append(
                f"{ticker} is trading with **bullish bias** as the long-term uptrend remains intact. "
                f"Although volume appears {volume_trend.lower()} recently, the price is firmly "
                f"{price_vs_sma_l} the long-term moving average. {rsi_interpretation.capitalize()} "
                f"suggests the uptrend still has momentum, despite the recent volume decline."
            )
            
            stories.append(
                f"The bullish technical setup in {ticker} persists with the price maintaining its position "
                f"{price_vs_sma_s} the short-term average. The {volume_trend.lower()} volume is noteworthy, "
                f"but as long as the price stays {price_vs_sma_l} the long-term average, the overall "
                f"uptrend bias remains in effect."
            )
    
    else:  # Bearish
        if volume_trend == "Increasing":
            stories.append(
                f"{ticker} is exhibiting **bearish pressure** with {volume_strength} selling volume "
                f"reinforcing the downside. The stock has broken below its short-term moving average and is "
                f"trading {price_vs_sma_l} the long-term average, signaling weakness. The combination of "
                f"{volume_strength} volume and declining prices suggests renewed selling interest."
            )
            
            stories.append(
                f"Deteriorating conditions in {ticker} are evident as rising volume accompanies the downtrend. "
                f"The price has fallen {price_vs_sma_l} the key long-term average, which often marks the "
                f"transition from uptrend to downtrend. With {rsi_interpretation}, further weakness may unfold."
            )
        else:
            stories.append(
                f"{ticker} remains under **bearish pressure** despite declining volume. The stock is trading "
                f"{price_vs_sma_l} both short and long-term moving averages, confirming the downtrend. "
                f"The reduced selling volume could indicate consolidation before the next downleg, or a "
                f"temporary pause in the decline."
            )
            
            stories.append(
                f"The bearish structure in {ticker} persists, with the price firmly below the long-term "
                f"moving average. Although sellers appear to be taking a breather (lower volume), the "
                f"downtrend remains the primary structure. Any recovery move should be treated with caution "
                f"until the stock reclaims the long-term average."
            )
    
    # Add confidence note
    confidence_note = (
        f"**Confidence Level: {confidence}%** — This analysis is based on technical indicators and "
        f"recent price action. Market conditions can change rapidly."
    )
    
    # Select a random story for variety
    import random
    selected_story = random.choice(stories)
    
    return f"{selected_story}\n\n{confidence_note}"

# ======================================
# Market Trend Detector
# ======================================

def detect_market_trend(df_data, sma_short_val, sma_long_val):
    """Detect market trend (Bullish or Bearish) with confidence score"""
    
    if len(df_data) < 10:
        return "Bearish", 50
    
    latest = df_data.iloc[-1]
    recent = df_data.iloc[-20:]  # Last 20 periods for trend analysis
    
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0
    
    # 1. SMA Alignment (40% weight)
    if latest['SMA_Short'] > latest['SMA_Long']:
        bullish_signals += 0.40
    else:
        bearish_signals += 0.40
    total_signals += 0.40
    
    # 2. Trend Slope (30% weight)
    recent_prices = recent['Close'].values
    x = np.arange(len(recent_prices))
    z = np.polyfit(x, recent_prices, 1)
    slope = z[0]
    
    if slope > 0:
        bullish_signals += 0.30
    else:
        bearish_signals += 0.30
    total_signals += 0.30
    
    # 3. Price above Long SMA (20% weight)
    if latest['Close'] > latest['SMA_Long']:
        bullish_signals += 0.20
    else:
        bearish_signals += 0.20
    total_signals += 0.20
    
    # 4. RSI Check - neutral/positive (10% weight)
    rsi_latest = latest['RSI'] if 'RSI' in latest.index else 50
    if rsi_latest > 40:  # More lenient threshold
        bullish_signals += 0.10
    else:
        bearish_signals += 0.10
    total_signals += 0.10
    
    # Calculate confidence
    if bullish_signals > bearish_signals:
        trend = "Bullish"
        confidence = int((bullish_signals / total_signals) * 100) if total_signals > 0 else 50
    else:
        trend = "Bearish"
        confidence = int((bearish_signals / total_signals) * 100) if total_signals > 0 else 50
    
    return trend, max(confidence, 50)

# Display Market Trend Detector
if not df.empty:
    trend, confidence = detect_market_trend(df, sma_short, sma_long)
    st.subheader(f"📊 Market Trend: {trend} ({confidence}%)")
    
    # Generate and display AI Market Story
    st.markdown("---")
    st.subheader("📖 AI Market Story Generator")
    market_story = generate_market_story(df, ticker, trend, confidence, sma_short, sma_long)
    st.markdown(market_story)

# ======================================
# Technical Analysis Charts
# ======================================

st.markdown("---")
st.subheader("📊 Technical Analysis Charts")

# Price + SMA Chart
# --------------------------------
st.subheader("Price & Moving Averages")

fig1, ax1 = plt.subplots(figsize=(12,5))

ax1.plot(df['Close'],label="Close Price")
ax1.plot(df['SMA_Short'],label=f"SMA {sma_short}")
ax1.plot(df['SMA_Long'],label=f"SMA {sma_long}")

ax1.set_title("Stock Price with Moving Averages")
ax1.legend()

st.pyplot(fig1)

# -------------------------------
# RSI Chart
# -------------------------------
st.subheader("RSI Indicator")

fig2, ax2 = plt.subplots(figsize=(12,4))

ax2.plot(df['RSI'],label="RSI")
ax2.axhline(70)
ax2.axhline(30)

ax2.set_title("Relative Strength Index")
ax2.legend()

st.pyplot(fig2)

# -------------------------------
# LSTM Prediction Chart
# -------------------------------
st.subheader("LSTM Price Prediction")

fig3, ax3 = plt.subplots(figsize=(12,5))

ax3.plot(df_lstm['Close'],label="Actual Price")
ax3.plot(df_lstm['LSTM_Prediction'],label="Predicted Price")

ax3.set_title("Actual vs LSTM Predicted Price")
ax3.legend()

st.pyplot(fig3)

# -------------------------------
# Multi-Day Future Price Prediction
# ----------------------------------
st.subheader("📈 Multi-Day Price Forecast")

# Number of days to predict
forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)

# Generate predictions for multiple days
last_60 = df[['Close']].values[-60:]
scaled_last = scaler.transform(last_60)

future_predictions = []

# Initialize sequence as 1D array of last 60 scaled prices
current_sequence = scaled_last.flatten().copy()

for day in range(forecast_days):
    # Reshape to (1, 60, 1) for model input
    X_test = current_sequence[-60:].reshape(1, 60, 1)
    next_pred = model.predict(X_test, verbose=0)
    next_pred_value = next_pred[0][0]
    future_predictions.append(next_pred_value)
    
    # Update sequence: append new prediction and keep last 60
    current_sequence = np.append(current_sequence, next_pred_value)

# Inverse transform predictions
future_predictions_array = np.array(future_predictions).reshape(-1, 1)
future_predictions_unscaled = scaler.inverse_transform(future_predictions_array)

# Create date range for predictions
last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

# Create prediction dataframe
df_forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions_unscaled.flatten()
})

# Display metrics for next day
col1, col2, col3 = st.columns(3)

with col1:
    next_day_price = future_predictions_unscaled[0][0]
    current_price = df['Close'].iloc[-1]
    price_change = next_day_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    st.metric(
        "Next Day Prediction",
        f"${next_day_price:.2f}",
        f"${price_change:+.2f} ({price_change_pct:+.2f}%)"
    )

with col2:
    min_pred = future_predictions_unscaled.min()
    st.metric(
        f"{forecast_days}-Day Low",
        f"${min_pred:.2f}"
    )

with col3:
    max_pred = future_predictions_unscaled.max()
    st.metric(
        f"{forecast_days}-Day High",
        f"${max_pred:.2f}"
    )

# Visualization
st.subheader(f"Price Forecast ({forecast_days} Days)")

fig_forecast, ax_forecast = plt.subplots(figsize=(14, 6))

# Plot historical data (last 60 days)
last_60_dates = df['Date'].tail(60)
last_60_prices = df['Close'].tail(60)

ax_forecast.plot(last_60_dates, last_60_prices, label='Historical Price', linewidth=2.5, color='blue')

# Connect forecast to current price
last_date = pd.Timestamp(df['Date'].max())
current_price = df['Close'].iloc[-1]

# Create continuous line from last historical point through predictions
# Convert future_dates to pandas DatetimeIndex for consistency
future_dates_index = pd.DatetimeIndex(future_dates)
connection_dates = pd.DatetimeIndex(list(df['Date'].tail(1)) + list(future_dates_index))
connection_prices = np.concatenate([
    [current_price],
    future_predictions_unscaled.flatten()
])

ax_forecast.plot(connection_dates, connection_prices, label='Predicted Price', 
                linewidth=2.5, color='red', linestyle='', marker='_', markersize=5)

# Add a vertical line separator
ax_forecast.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Add annotations for clarity
ax_forecast.text(last_date, current_price, '  Current', fontsize=9, va='center', color='black', fontweight='bold')

# Formatting
ax_forecast.set_title(f'{ticker} - {forecast_days} Day Price Forecast', fontsize=14, fontweight='bold')
ax_forecast.set_xlabel('Date', fontsize=11)
ax_forecast.set_ylabel('Price ($)', fontsize=11)
ax_forecast.legend(loc='best', fontsize=10)
ax_forecast.grid(True, alpha=0.3)

# Format x-axis dates properly
import matplotlib.dates as mdates
ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig_forecast.autofmt_xdate(rotation=45, ha='right')

st.pyplot(fig_forecast)

# Display forecast table
st.subheader("Forecast Details")

forecast_display = pd.DataFrame({
    'Date': df_forecast['Date'].dt.strftime('%Y-%m-%d'),
    'Predicted Price': [f"${p:.2f}" for p in df_forecast['Predicted_Price']],
    'Change from Current': [f"{((p - current_price) / current_price * 100):+.2f}%" 
                           for p in df_forecast['Predicted_Price']],
})

st.dataframe(forecast_display, use_container_width=True, hide_index=True)

# Prediction confidence note
st.info("""
📌 **Forecast Accuracy Note:**
- LSTM predictions are based on historical patterns
- Accuracy decreases for longer forecast periods
- Real-world events can affect stock prices unpredictably
- Use predictions as a reference, not a guarantee
- Combine with technical analysis and news for better decisions
""")

# -------------------------------
# Backtesting Results
# -------------------------------
st.subheader("Strategy Backtesting")

fig4, ax4 = plt.subplots(figsize=(12,5))

ax4.plot(cumulative_market,label="Market Returns")
ax4.plot(cumulative_strategy,label="Strategy Returns")

ax4.set_title("Cumulative Returns Comparison")
ax4.legend()

st.pyplot(fig4)

# -------------------------------
# Performance Metrics
# -------------------------------
st.subheader("Performance Metrics")

total_market_return = (cumulative_market.iloc[-1]-1)*100
total_strategy_return = (cumulative_strategy.iloc[-1]-1)*100

col1,col2 = st.columns(2)

col1.metric("Market Return (%)",f"{total_market_return:.2f}")
col2.metric("Strategy Return (%)",f"{total_strategy_return:.2f}")

# -------------------------------
# Buy / Sell Signal
# -------------------------------
st.subheader("Latest Trading Signal")

latest_signal = df['Signal'].iloc[-1]

if latest_signal == 1:
    st.success("BUY Signal")

elif latest_signal == -1:
    st.error("SELL Signal")

else:
    st.warning("HOLD")
