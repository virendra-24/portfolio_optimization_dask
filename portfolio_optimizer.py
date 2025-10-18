import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Dask imports for Distributed Computing
import dask
import dask.dataframe as dd
import dask.array as da
from dask import delayed
import dask.bag as db

# Page Configuration
st.set_page_config(
    page_title="Advanced Portfolio Optimizer with Dask",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Header
st.title("⚡ Advanced Portfolio Optimizer using Dask (Big Data Analytics)")
st.markdown("### Distributed Computing for Large-Scale Financial Data Processing & Multi-Timeframe Analysis")
st.markdown("---")

# Initialize Dask Client
@st.cache_resource
def init_dask():
    """Initialize Dask Client for distributed computing"""
    try:
        from dask.distributed import Client
        client = Client(processes=False, n_workers=4, threads_per_worker=2, memory_limit='2GB')
        st.success("✅ Dask Client initialized with 4 workers!")
        return client
    except Exception as e:
        st.warning(f"⚠️ Using Dask with threaded scheduler: {e}")
        dask.config.set(scheduler='threads', num_workers=4)
        return None

# Sidebar - Input Parameters
st.sidebar.header("🧱 Portfolio Configuration")

# Portfolio Input Mode Selection
st.sidebar.markdown("### 📊 Portfolio Input Mode")
input_mode = st.sidebar.radio(
    "Choose how to define your portfolio:",
    ["Quick Select (Auto Weights)", "Custom Weights (Manual)"],
    help="Quick Select uses equal weights, Custom Weights lets you specify exact allocations"
)

st.sidebar.markdown("### Select Assets")

# Stock Selection - US and Indian Stocks
us_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'WIPRO.NS',
                 'TATAMOTORS.NS', 'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS']
available_tickers = us_stocks + indian_stocks

st.sidebar.markdown("#### 📋 Select from Popular Stocks")

# Add category selection
stock_category = st.sidebar.radio(
    "Stock Category:",
    ["🌐 Global Mix", "🇺🇸 US Stocks", "🇮🇳 Indian Stocks"],
    horizontal=True
)

if stock_category == "🇺🇸 US Stocks":
    display_tickers = us_stocks
    default_selection = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
elif stock_category == "🇮🇳 Indian Stocks":
    display_tickers = indian_stocks
    default_selection = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
else:
    display_tickers = available_tickers
    default_selection = ['AAPL', 'MSFT', 'RELIANCE.NS', 'TCS.NS']

selected_predefined = st.sidebar.multiselect(
    "Choose stocks:",
    display_tickers,
    default=default_selection
)

st.sidebar.markdown("#### 🆕 Add Custom Stocks")
custom_tickers_input = st.sidebar.text_input(
    "Enter custom stock tickers (comma-separated):",
    placeholder="e.g., BTC-USD, ETH-USD, GOLD, TLT",
    help="Enter any valid Yahoo Finance ticker"
)

# Combine selections
selected_tickers = selected_predefined.copy()
if custom_tickers_input:
    custom_tickers = [ticker.strip().upper() for ticker in custom_tickers_input.split(',') if ticker.strip()]
    selected_tickers.extend(custom_tickers)
    seen = set()
    selected_tickers = [ticker for ticker in selected_tickers if not (ticker in seen or seen.add(ticker))]

# Portfolio Weights Input
portfolio_weights = {}
if selected_tickers and input_mode == "Custom Weights (Manual)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ Portfolio Weights")
    st.sidebar.info("Enter weights as percentages (must sum to 100%)")
    
    total_weight = 0
    for ticker in selected_tickers:
        weight = st.sidebar.number_input(
            f"{ticker} Weight (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0/len(selected_tickers),
            step=1.0,
            key=f"weight_{ticker}"
        )
        portfolio_weights[ticker] = weight
        total_weight += weight
    
    # Display total weight
    if abs(total_weight - 100.0) > 0.01:
        st.sidebar.error(f"⚠️ Total weight: {total_weight:.1f}% (must equal 100%)")
    else:
        st.sidebar.success(f"✅ Total weight: {total_weight:.1f}%")

# Display selected tickers
if selected_tickers:
    st.sidebar.markdown("#### ✅ Selected Stocks")
    st.sidebar.write(f"**Total:** {len(selected_tickers)} stocks")
    st.sidebar.write(", ".join(selected_tickers))

# Date Range
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2015, 1, 1)
)
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now()
)

# Simulation Parameters
st.sidebar.markdown("### 🎲 Simulation Parameters")
num_simulations = st.sidebar.slider(
    "Monte Carlo Simulations",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000,
    help="More simulations = better optimization but slower processing"
)

rf_rate = st.sidebar.number_input(
    "Risk-Free Rate (Annualized %)",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
) / 100

# Investment Amount for Projection
st.sidebar.markdown("### 💰 Investment Amount")

# Currency selection
currency = st.sidebar.radio(
    "Currency:",
    ["₹ INR (Indian Rupees)", "$ USD (US Dollars)"],
    horizontal=True
)

if currency == "₹ INR (Indian Rupees)":
    investment_amount = st.sidebar.number_input(
        "Initial Investment (₹)",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=100000,
        help="Enter amount in Indian Rupees"
    )
    currency_symbol = "₹"
    usd_rate = 83.0  # Approximate conversion rate
    investment_usd = investment_amount / usd_rate
else:
    investment_amount = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Enter amount in US Dollars"
    )
    currency_symbol = "$"
    usd_rate = 1.0
    investment_usd = investment_amount

# Action Buttons
st.sidebar.markdown("---")
fetch_data_btn = st.sidebar.button("📥 Fetch Data with Dask", use_container_width=True, type="primary")
run_simulation_btn = st.sidebar.button("▶️ Run Optimization", use_container_width=True, type="secondary")

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'dask_client' not in st.session_state:
    st.session_state.dask_client = None

# BDA Concepts Highlight
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 BDA Features (Dask)")
st.sidebar.success("""
✅ **Parallel Processing**: Multi-worker execution  
✅ **Lazy Evaluation**: Deferred computation  
✅ **Task Scheduling**: Intelligent workload distribution  
✅ **Memory Management**: Efficient data handling  
✅ **Scalability**: Handles millions of simulations  
✅ **Distributed DataFrames**: Big data operations  
""")

# Main Content
if len(selected_tickers) < 2:
    st.warning("⚠️ Please select at least 2 stocks to proceed.")
    st.stop()

# Validate weights if custom mode
if input_mode == "Custom Weights (Manual)":
    total_weight = sum(portfolio_weights.values())
    if abs(total_weight - 100.0) > 0.01:
        st.error("❌ Portfolio weights must sum to 100%. Please adjust weights in the sidebar.")
        st.stop()

# Dask initialization
with st.expander("⚡ Dask Cluster Status", expanded=False):
    if st.session_state.dask_client is None:
        with st.spinner("Initializing Dask cluster..."):
            client = init_dask()
            st.session_state.dask_client = client
        
        st.success("✅ Dask processing framework active!")
        st.code("""
Dask Configuration:
- Workers: 4 parallel workers
- Framework: Distributed DataFrames + Delayed
- Scheduler: Threaded/Distributed
- Capabilities: Parallel Monte Carlo, Lazy Evaluation
        """)

# Function to fetch stock data
@st.cache_data
def fetch_stock_data_dask(tickers, start, end):
    """Fetch historical stock data using yFinance API and convert to Dask DataFrame"""
    try:
        valid_tickers = [t for t in tickers if len(t) > 0 and all(c.isalnum() or c in ['-', '=', '.'] for c in t)]
        
        if not valid_tickers:
            return None, None
        
        raw_data = yf.download(valid_tickers, start=start, end=end, progress=False)
        
        if raw_data.empty:
            return None, None
        
        # Extract prices
        if 'Adj Close' in raw_data.columns.get_level_values(0):
            data = raw_data['Adj Close'].copy()
        else:
            data = raw_data['Close'].copy()
        
        # Simplify columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1) if len(data.columns.levels) > 1 else data.columns.get_level_values(0)
        
        # Handle single ticker case
        if len(valid_tickers) == 1 and isinstance(data, pd.Series):
            data = data.to_frame(valid_tickers[0])
        
        data = data.dropna()
        
        if data.empty:
            return None, None
        
        # Convert to long format for Dask
        data_reset = data.reset_index()
        date_col = data_reset.columns[0]
        data_reset = data_reset.rename(columns={date_col: 'Date'})
        data_long = data_reset.melt(id_vars=['Date'], var_name='Ticker', value_name='Price')
        
        return data, data_long
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# Function to process data with Dask
def process_data_with_dask(data_long):
    """Process financial data using Dask distributed computing"""
    try:
        dask_df = dd.from_pandas(data_long, npartitions=8)
        
        def calculate_returns_chunk(chunk):
            chunk = chunk.sort_values(['Ticker', 'Date'])
            chunk['Prev_Price'] = chunk.groupby('Ticker')['Price'].shift(1)
            chunk['Return'] = (chunk['Price'] - chunk['Prev_Price']) / chunk['Prev_Price']
            return chunk
        
        dask_df = dask_df.map_partitions(
            calculate_returns_chunk,
            meta={
                'Date': 'datetime64[ns]',
                'Ticker': 'object',
                'Price': 'float64',
                'Prev_Price': 'float64',
                'Return': 'float64'
            }
        )
        
        dask_df = dask_df.dropna(subset=['Return'])
        return dask_df
    except Exception as e:
        st.error(f"Error in Dask processing: {e}")
        return None

# Function to calculate statistics
def calculate_statistics_dask(_dask_df, tickers):
    """Calculate statistics using Dask aggregations"""
    try:
        stats_df = _dask_df.groupby('Ticker').agg({
            'Return': ['mean', 'std', 'count']
        }).compute()
        
        stats_df.columns = ['Mean_Return', 'Std_Return', 'Count']
        stats_df = stats_df.reset_index()
        stats_df['Annual_Return'] = stats_df['Mean_Return'] * 252
        stats_df['Annual_Volatility'] = stats_df['Std_Return'] * np.sqrt(252)
        
        return stats_df
    except Exception as e:
        st.error(f"Error calculating statistics: {e}")
        return None

# Function to calculate correlation
def calculate_correlation_dask(_dask_df, tickers):
    """Calculate correlation matrix using Dask"""
    try:
        returns_pivot = _dask_df[['Date', 'Ticker', 'Return']].compute()
        returns_pivot = returns_pivot.pivot_table(
            index='Date', 
            columns='Ticker', 
            values='Return'
        )
        correlation_matrix = returns_pivot.corr()
        return correlation_matrix, returns_pivot
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")
        return None, None

# Calculate current portfolio metrics
def calculate_current_portfolio_metrics(returns_df, weights_dict, rf_rate):
    """Calculate metrics for current portfolio"""
    try:
        # Convert weights to array in correct order
        tickers = returns_df.columns.tolist()
        weights = np.array([weights_dict.get(t, 0) / 100 for t in tickers])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        port_return = np.sum(weights * mean_returns) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = (port_return - rf_rate) / port_volatility if port_volatility > 0 else 0
        
        # Calculate VaR and CVaR
        portfolio_returns = returns_df.dot(weights)
        if len(portfolio_returns) > 10:
            var = np.percentile(portfolio_returns, 5)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            var_annual = var * np.sqrt(252)
            cvar_annual = cvar * np.sqrt(252)
        else:
            var_annual = np.nan
            cvar_annual = np.nan
        
        return {
            'Return': port_return,
            'Volatility': port_volatility,
            'Sharpe': sharpe,
            'VaR': var_annual,
            'CVaR': cvar_annual,
            'Weights': weights
        }
    except Exception as e:
        st.error(f"Error calculating current portfolio: {e}")
        return None

# Monte Carlo Simulation with Dask
def monte_carlo_simulation_dask(returns_df, tickers, num_sims, rf_rate):
    """Run Monte Carlo simulation using Dask for distributed parallel processing"""
    try:
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        @delayed
        def simulate_portfolio(seed):
            np.random.seed(seed)
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            
            port_return = np.sum(weights * mean_returns) * 252
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = (port_return - rf_rate) / port_volatility if port_volatility > 0 else 0
            
            try:
                portfolio_returns = returns_df.dot(weights)
                if len(portfolio_returns) > 10 and not np.any(np.isnan(portfolio_returns)):
                    var = np.nanpercentile(portfolio_returns, 5)
                    tail_returns = portfolio_returns[portfolio_returns <= var]
                    cvar = tail_returns.mean() if len(tail_returns) > 0 else var * 1.1
                    var_annual = var * np.sqrt(252)
                    cvar_annual = cvar * np.sqrt(252)
                else:
                    var_annual = port_return - 1.645 * port_volatility
                    cvar_annual = port_return - 2.063 * port_volatility
            except:
                var_annual = port_return * 0.8
                cvar_annual = port_return * 0.7
            
            return (float(port_return), float(port_volatility), float(sharpe), 
                    float(var_annual), float(cvar_annual), *[float(w) for w in weights])
        
        delayed_results = [simulate_portfolio(i) for i in range(num_sims)]
        results = dask.compute(*delayed_results)
        
        return list(results)
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {e}")
        return None

# Project portfolio returns over multiple years
def project_portfolio_returns(annual_return, annual_volatility, years, initial_investment):
    """Project portfolio value over multiple years with confidence intervals"""
    projections = {}
    for year in years:
        # Expected value (mean scenario)
        expected_value = initial_investment * ((1 + annual_return) ** year)
        
        # Confidence intervals using normal distribution
        std_dev_year = annual_volatility * np.sqrt(year)
        
        # 95% confidence interval
        upper_95 = initial_investment * np.exp((annual_return - 0.5 * annual_volatility**2) * year + 1.96 * std_dev_year)
        lower_95 = initial_investment * np.exp((annual_return - 0.5 * annual_volatility**2) * year - 1.96 * std_dev_year)
        
        # 68% confidence interval (1 std dev)
        upper_68 = initial_investment * np.exp((annual_return - 0.5 * annual_volatility**2) * year + std_dev_year)
        lower_68 = initial_investment * np.exp((annual_return - 0.5 * annual_volatility**2) * year - std_dev_year)
        
        projections[year] = {
            'Expected': expected_value,
            'Upper_95': upper_95,
            'Lower_95': lower_95,
            'Upper_68': upper_68,
            'Lower_68': lower_68,
            'Return_Pct': (expected_value - initial_investment) / initial_investment * 100
        }
    
    return projections

# Fetch Data Section
if fetch_data_btn:
    with st.spinner("🔄 Fetching stock data and processing with Dask..."):
        stock_data, data_long = fetch_stock_data_dask(selected_tickers, start_date, end_date)
        
        if stock_data is not None and not stock_data.empty:
            available_tickers = stock_data.columns.tolist()
            
            dask_df = process_data_with_dask(data_long)
            
            if dask_df is not None:
                stats_pandas = calculate_statistics_dask(dask_df, available_tickers)
                correlation_matrix, returns_df = calculate_correlation_dask(dask_df, available_tickers)
                
                if stats_pandas is not None and correlation_matrix is not None:
                    st.session_state.stock_data = stock_data
                    st.session_state.dask_df = dask_df
                    st.session_state.stats_pandas = stats_pandas
                    st.session_state.correlation_matrix = correlation_matrix
                    st.session_state.returns_df = returns_df
                    st.session_state.available_tickers = available_tickers
                    st.session_state.data_fetched = True
                    
                    # Calculate current portfolio metrics if custom weights
                    if input_mode == "Custom Weights (Manual)":
                        # Filter weights for available tickers
                        available_weights = {t: portfolio_weights[t] for t in available_tickers if t in portfolio_weights}
                        current_metrics = calculate_current_portfolio_metrics(returns_df, available_weights, rf_rate)
                        st.session_state.current_portfolio = current_metrics
                        st.session_state.current_weights = available_weights
                    
                    st.success(f"✅ Successfully processed {len(available_tickers)} stocks using Dask!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records Processed", f"{dask_df.shape[0].compute():,}")
                    with col2:
                        st.metric("Dask Partitions", dask_df.npartitions)
                    with col3:
                        st.metric("Assets Analyzed", len(available_tickers))

# Display Data Section
if st.session_state.data_fetched:
    st.markdown("---")
    st.markdown("## 📊 Data Analysis (Dask Processed)")
    
    available_tickers = st.session_state.available_tickers
    
    tab1, tab2, tab3 = st.tabs(["📈 Price Trends", "📊 Statistics", "🔗 Correlations"])
    
    with tab1:
        normalized_data = st.session_state.stock_data / st.session_state.stock_data.iloc[0] * 100
        
        fig_prices = go.Figure()
        for ticker in available_tickers:
            fig_prices.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[ticker],
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
        
        fig_prices.update_layout(
            title="Normalized Stock Prices (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode='x unified',
            height=450
        )
        st.plotly_chart(fig_prices, use_container_width=True)
    
    with tab2:
        if st.session_state.stats_pandas is not None:
            stats_display = st.session_state.stats_pandas[['Ticker', 'Annual_Return', 'Annual_Volatility']].copy()
            stats_display['Annual_Return'] = (stats_display['Annual_Return'] * 100).round(2)
            stats_display['Annual_Volatility'] = (stats_display['Annual_Volatility'] * 100).round(2)
            stats_display.columns = ['Ticker', 'Return (%)', 'Volatility (%)']
            
            st.dataframe(stats_display, use_container_width=True, hide_index=True)
    
    with tab3:
        if st.session_state.correlation_matrix is not None:
            correlation_matrix = st.session_state.correlation_matrix
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Asset Correlation Matrix",
                height=450
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# Run Simulation
if run_simulation_btn and st.session_state.data_fetched:
    st.markdown("---")
    st.markdown("## ⚡ Portfolio Optimization (Dask Distributed)")
    
    with st.spinner(f"🔄 Running {num_simulations:,} simulations using Dask..."):
        returns_df = st.session_state.returns_df
        available_tickers = st.session_state.available_tickers
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("⚡ Distributing tasks across Dask workers...")
        
        progress_bar.progress(0.3)
        results = monte_carlo_simulation_dask(
            returns_df, available_tickers, num_simulations, rf_rate
        )
        
        if results:
            progress_bar.progress(0.8)
            status_text.text("⚡ Collecting results...")
            
            columns = ['Return', 'Volatility', 'Sharpe', 'VaR', 'CVaR'] + available_tickers
            results_df = pd.DataFrame(results, columns=columns)
            
            valid_results = results_df.dropna(subset=['VaR', 'CVaR'])
            if len(valid_results) > 0:
                results_df = valid_results
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed {num_simulations:,} simulations!")
            
            if len(results_df) > 0:
                max_sharpe_idx = results_df['Sharpe'].idxmax()
                optimal_portfolio = results_df.loc[max_sharpe_idx]
                
                st.session_state.results_df = results_df
                st.session_state.optimal_portfolio = optimal_portfolio
                st.session_state.simulation_run = True
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Simulations", f"{len(results_df):,}")
                with col2:
                    st.metric("Mode", "Parallel Dask")
                with col3:
                    st.metric("Framework", "Distributed")
                with col4:
                    st.metric("Best Sharpe", f"{optimal_portfolio['Sharpe']:.3f}")

# Display Results
if st.session_state.simulation_run:
    st.markdown("---")
    st.markdown("## 📈 Optimization Results")
    
    results_df = st.session_state.results_df
    optimal_portfolio = st.session_state.optimal_portfolio
    available_tickers = st.session_state.available_tickers
    
    # Show comparison if custom weights were provided
    if input_mode == "Custom Weights (Manual)" and 'current_portfolio' in st.session_state:
        st.markdown("### 🔄 Portfolio Comparison: Current vs Optimized")
        
        current = st.session_state.current_portfolio
        optimal = optimal_portfolio
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Expected Return**")
            st.metric(
                "Current",
                f"{current['Return']*100:.2f}%"
            )
            st.metric(
                "Optimized",
                f"{optimal['Return']*100:.2f}%",
                delta=f"{(optimal['Return']-current['Return'])*100:.2f}%"
            )
        
        with col2:
            st.markdown("**Volatility**")
            st.metric(
                "Current",
                f"{current['Volatility']*100:.2f}%"
            )
            st.metric(
                "Optimized",
                f"{optimal['Volatility']*100:.2f}%",
                delta=f"{(optimal['Volatility']-current['Volatility'])*100:.2f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.markdown("**Sharpe Ratio**")
            st.metric(
                "Current",
                f"{current['Sharpe']:.3f}"
            )
            st.metric(
                "Optimized",
                f"{optimal['Sharpe']:.3f}",
                delta=f"{optimal['Sharpe']-current['Sharpe']:.3f}"
            )
        
        with col4:
            st.markdown("**95% VaR**")
            if not pd.isna(current['VaR']):
                st.metric(
                    "Current",
                    f"{current['VaR']*100:.2f}%"
                )
            if not pd.isna(optimal['VaR']):
                st.metric(
                    "Optimized",
                    f"{optimal['VaR']*100:.2f}%"
                )
    
    # Efficient Frontier
    st.markdown("### 📊 Efficient Frontier")
    
    fig_ef = go.Figure()
    
    fig_ef.add_trace(go.Scatter(
        x=results_df['Volatility'],
        y=results_df['Return'],
        mode='markers',
        marker=dict(
            size=3,
            color=results_df['Sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            opacity=0.6
        ),
        name='Simulated Portfolios',
        hovertemplate='<b>Return:</b> %{y:.2%}<br><b>Volatility:</b> %{x:.2%}<br><extra></extra>'
    ))
    
    # Add current portfolio if exists
    if input_mode == "Custom Weights (Manual)" and 'current_portfolio' in st.session_state:
        current = st.session_state.current_portfolio
        fig_ef.add_trace(go.Scatter(
            x=[current['Volatility']],
            y=[current['Return']],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='circle', line=dict(width=2, color='white')),
            name='Your Current Portfolio'
        ))
    
    # Add optimal portfolio
    fig_ef.add_trace(go.Scatter(
        x=[optimal_portfolio['Volatility']],
        y=[optimal_portfolio['Return']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='white')),
        name='Optimal Portfolio'
    ))
    
    fig_ef.update_layout(
        title="Risk-Return Profile (Dask Monte Carlo)",
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_ef, use_container_width=True)
    
    # Multi-Timeframe Projections
    st.markdown("---")
    st.markdown("## 📅 Multi-Timeframe Portfolio Projections")
    
    projection_years = [1, 2, 3, 5, 10]
    
    # Calculate projections for both portfolios
    optimal_projections = project_portfolio_returns(
        optimal_portfolio['Return'],
        optimal_portfolio['Volatility'],
        projection_years,
        investment_amount
    )
    
    if input_mode == "Custom Weights (Manual)" and 'current_portfolio' in st.session_state:
        current_projections = project_portfolio_returns(
            st.session_state.current_portfolio['Return'],
            st.session_state.current_portfolio['Volatility'],
            projection_years,
            investment_amount
        )
    else:
        current_projections = None
    
    # Display projection table
    st.markdown("### 💵 Expected Portfolio Value Over Time")
    
    projection_data = []
    for year in projection_years:
        row = {
            'Years': year,
            'Optimized Value': f"{currency_symbol}{optimal_projections[year]['Expected']:,.2f}",
            'Optimized Return': f"{optimal_projections[year]['Return_Pct']:.1f}%",
            'Optimized Range (95%)': f"{currency_symbol}{optimal_projections[year]['Lower_95']:,.0f} - {currency_symbol}{optimal_projections[year]['Upper_95']:,.0f}"
        }
        
        if current_projections:
            row['Current Value'] = f"{currency_symbol}{current_projections[year]['Expected']:,.2f}"
            row['Current Return'] = f"{current_projections[year]['Return_Pct']:.1f}%"
            row['Improvement'] = f"{currency_symbol}{optimal_projections[year]['Expected'] - current_projections[year]['Expected']:,.2f}"
        
        projection_data.append(row)
    
    projection_df = pd.DataFrame(projection_data)
    st.dataframe(projection_df, use_container_width=True, hide_index=True)
    
    # Visualization of projections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Portfolio Growth Projection")
        
        fig_growth = go.Figure()
        
        years_plot = [0] + projection_years
        optimal_values = [investment_amount] + [optimal_projections[y]['Expected'] for y in projection_years]
        
        fig_growth.add_trace(go.Scatter(
            x=years_plot,
            y=optimal_values,
            mode='lines+markers',
            name='Optimized Portfolio',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        
        if current_projections:
            current_values = [investment_amount] + [current_projections[y]['Expected'] for y in projection_years]
            fig_growth.add_trace(go.Scatter(
                x=years_plot,
                y=current_values,
                mode='lines+markers',
                name='Current Portfolio',
                line=dict(color='blue', width=3, dash='dash'),
                marker=dict(size=10)
            ))
        
        fig_growth.update_layout(
            title="Expected Portfolio Value Growth",
            xaxis_title="Years",
            yaxis_title=f"Portfolio Value ({currency_symbol})",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Cumulative Return Comparison")
        
        fig_return = go.Figure()
        
        optimal_returns = [0] + [optimal_projections[y]['Return_Pct'] for y in projection_years]
        
        fig_return.add_trace(go.Bar(
            x=years_plot,
            y=optimal_returns,
            name='Optimized Portfolio',
            marker_color='green'
        ))
        
        if current_projections:
            current_returns = [0] + [current_projections[y]['Return_Pct'] for y in projection_years]
            fig_return.add_trace(go.Bar(
                x=years_plot,
                y=current_returns,
                name='Current Portfolio',
                marker_color='blue'
            ))
        
        fig_return.update_layout(
            title="Cumulative Return (%)",
            xaxis_title="Years",
            yaxis_title="Return (%)",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig_return, use_container_width=True)
    
    # Confidence intervals for 5-year projection
    st.markdown("#### 🎯 5-Year Projection with Confidence Intervals")
    
    five_year = optimal_projections[5]
    
    fig_ci = go.Figure()
    
    scenarios = ['Pessimistic\n(5%)', 'Conservative\n(16%)', 'Expected\n(50%)', 'Optimistic\n(84%)', 'Best Case\n(95%)']
    values = [
        five_year['Lower_95'],
        five_year['Lower_68'],
        five_year['Expected'],
        five_year['Upper_68'],
        five_year['Upper_95']
    ]
    colors = ['red', 'orange', 'green', 'lightgreen', 'darkgreen']
    
    fig_ci.add_trace(go.Bar(
        x=scenarios,
        y=values,
        marker_color=colors,
        text=[f"{currency_symbol}{v:,.0f}" for v in values],
        textposition='auto'
    ))
    
    fig_ci.add_hline(
        y=investment_amount,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Investment"
    )
    
    fig_ci.update_layout(
        title="5-Year Portfolio Value Scenarios (Optimized Portfolio)",
        xaxis_title="Scenario",
        yaxis_title=f"Portfolio Value ({currency_symbol})",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_ci, use_container_width=True)
    
    # Key Insights
    st.markdown("### 🔑 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        five_yr_return = optimal_projections[5]['Return_Pct']
        st.info(f"""
        **5-Year Outlook**
        
        Expected Value: {currency_symbol}{optimal_projections[5]['Expected']:,.2f}
        
        Total Return: {five_yr_return:.1f}%
        
        Annualized: {optimal_portfolio['Return']*100:.2f}%
        """)
    
    with col2:
        if current_projections:
            improvement_5yr = optimal_projections[5]['Expected'] - current_projections[5]['Expected']
            improvement_pct = (improvement_5yr / current_projections[5]['Expected']) * 100
            st.success(f"""
            **Optimization Benefit**
            
            Additional Gain: {currency_symbol}{improvement_5yr:,.2f}
            
            Improvement: {improvement_pct:.1f}%
            
            Over 5 Years
            """)
        else:
            st.success(f"""
            **Risk-Adjusted Return**
            
            Sharpe Ratio: {optimal_portfolio['Sharpe']:.3f}
            
            Return/Risk: {optimal_portfolio['Return']/optimal_portfolio['Volatility']:.2f}
            
            Efficient Portfolio
            """)
    
    with col3:
        range_5yr = five_year['Upper_95'] - five_year['Lower_95']
        st.warning(f"""
        **Risk Assessment**
        
        95% Range: {currency_symbol}{range_5yr:,.0f}
        
        Volatility: {optimal_portfolio['Volatility']*100:.2f}%
        
        VaR (95%): {optimal_portfolio['VaR']*100:.2f}%
        """)
    
    # Optimal Portfolio Details
    st.markdown("---")
    st.markdown("## 📊 Optimal Portfolio Allocation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        weights_data = {ticker: optimal_portfolio[ticker] for ticker in available_tickers}
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(weights_data.keys()),
            values=list(weights_data.values()),
            hole=0.4,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_pie.update_layout(
            title="Optimized Asset Allocation",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = go.Figure()
        
        # Add optimized weights
        fig_bar.add_trace(go.Bar(
            x=list(weights_data.keys()),
            y=[v*100 for v in weights_data.values()],
            name='Optimized',
            marker_color='green',
            text=[f"{v*100:.1f}%" for v in weights_data.values()],
            textposition='auto'
        ))
        
        # Add current weights if available
        if input_mode == "Custom Weights (Manual)" and 'current_weights' in st.session_state:
            current_weights_data = st.session_state.current_weights
            fig_bar.add_trace(go.Bar(
                x=list(current_weights_data.keys()),
                y=list(current_weights_data.values()),
                name='Current',
                marker_color='blue',
                text=[f"{v:.1f}%" for v in current_weights_data.values()],
                textposition='auto'
            ))
        
        fig_bar.update_layout(
            title="Weight Comparison",
            xaxis_title="Asset",
            yaxis_title="Weight (%)",
            height=400,
            barmode='group'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed allocation table
    st.markdown("### 📋 Detailed Portfolio Allocation")
    
    weights_table_data = []
    for ticker in available_tickers:
        row = {
            'Ticker': ticker,
            'Optimized Weight (%)': f"{optimal_portfolio[ticker]*100:.2f}",
            'Optimized Amount': f"{currency_symbol}{optimal_portfolio[ticker]*investment_amount:,.2f}"
        }
        
        if input_mode == "Custom Weights (Manual)" and 'current_weights' in st.session_state:
            current_weight = st.session_state.current_weights.get(ticker, 0)
            row['Current Weight (%)'] = f"{current_weight:.2f}"
            row['Current Amount'] = f"{currency_symbol}{(current_weight/100)*investment_amount:,.2f}"
            row['Change'] = f"{(optimal_portfolio[ticker]*100 - current_weight):+.2f}%"
        
        weights_table_data.append(row)
    
    weights_table_df = pd.DataFrame(weights_table_data)
    weights_table_df = weights_table_df.sort_values('Optimized Weight (%)', ascending=False)
    
    st.dataframe(weights_table_df, use_container_width=True, hide_index=True)
    
    # Risk Metrics Dashboard
    st.markdown("---")
    st.markdown("## 🛡️ Risk Metrics Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Expected Return",
            f"{optimal_portfolio['Return']*100:.2f}%",
            delta=f"{(optimal_portfolio['Return']-rf_rate)*100:.2f}% vs RF"
        )
    
    with col2:
        st.metric(
            "Volatility",
            f"{optimal_portfolio['Volatility']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{optimal_portfolio['Sharpe']:.3f}"
        )
    
    with col4:
        if not pd.isna(optimal_portfolio['VaR']):
            st.metric(
                "VaR (95%)",
                f"{optimal_portfolio['VaR']*100:.2f}%",
                delta_color="inverse"
            )
        else:
            st.metric("VaR (95%)", "N/A")
    
    with col5:
        if not pd.isna(optimal_portfolio['CVaR']):
            st.metric(
                "CVaR (95%)",
                f"{optimal_portfolio['CVaR']*100:.2f}%",
                delta_color="inverse"
            )
        else:
            st.metric("CVaR (95%)", "N/A")
    
    # Return Distribution
    st.markdown("### 📊 Return Distribution Analysis")
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=results_df['Return']*100,
        nbinsx=50,
        name='Portfolio Returns',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig_hist.add_vline(
        x=optimal_portfolio['Return']*100,
        line_dash="solid",
        line_color="green",
        annotation_text="Optimal Return",
        annotation_position="top right"
    )
    
    if not pd.isna(optimal_portfolio['VaR']):
        fig_hist.add_vline(
            x=optimal_portfolio['VaR']*100,
            line_dash="dash",
            line_color="red",
            annotation_text="95% VaR",
            annotation_position="bottom right"
        )
    
    fig_hist.update_layout(
        title="Distribution of Simulated Portfolio Returns",
        xaxis_title="Annual Return (%)",
        yaxis_title="Frequency",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Dask Performance Metrics
    st.markdown("---")
    st.markdown("## ⚡ Big Data Analytics Performance (Dask)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Mode", "Parallel Dask")
    with col2:
        st.metric("Simulations", f"{len(results_df):,}")
    with col3:
        st.metric("Framework", "Distributed")
    with col4:
        st.metric("Data Partitions", st.session_state.dask_df.npartitions)
    
    st.info("""
    **Dask Big Data Analytics Features Used:**
    - ✅ **Parallel Monte Carlo Simulation**: Distributed task execution across workers
    - ✅ **Lazy Evaluation**: Deferred computation for efficiency
    - ✅ **DataFrames Partitioning**: Data split across 8 partitions for parallel processing
    - ✅ **Delayed Objects**: Task graph optimization for complex workflows
    - ✅ **Memory Management**: Efficient handling of large datasets
    - ✅ **Task Scheduling**: Intelligent workload distribution
    - ✅ **Scalability**: Can handle 50K+ simulations efficiently
    """)
    
    # Download Section
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_optimal = weights_table_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Allocation (CSV)",
            data=csv_optimal,
            file_name=f"optimal_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_projections = projection_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Projections (CSV)",
            data=csv_projections,
            file_name=f"portfolio_projections_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col3:
        csv_all = results_df.to_csv(index=False)
        st.download_button(
            label="🗃️ Download All Simulations (CSV)",
            data=csv_all,
            file_name=f"monte_carlo_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <h4>⚡ Advanced Portfolio Optimizer with Big Data Analytics</h4>
    <p><b>Technologies:</b> Dask Distributed Computing | Streamlit | Plotly | yFinance | NumPy | Pandas</p>
    <p><b>BDA Concepts:</b> Parallel Processing | Lazy Evaluation | Task Scheduling | Distributed DataFrames | Memory Efficiency</p>
    <p><b>Features:</b> Monte Carlo Simulation | Multi-Timeframe Projections | Risk Metrics | Portfolio Optimization</p>
    <hr style='width: 50%; margin: 20px auto;'>
    <p style='font-size: 12px;'>
        <b>Disclaimer:</b> This tool is for educational and informational purposes only. 
        Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.
    </p>
</div>
""", unsafe_allow_html=True)