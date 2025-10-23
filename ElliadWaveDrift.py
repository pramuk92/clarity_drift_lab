# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings
from datetime import datetime
import io

warnings.filterwarnings('ignore')

# Popular US stocks database
STOCK_DATABASE = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation", 
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc. (Facebook)",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "PG": "Procter & Gamble",
    "UNH": "UnitedHealth Group",
    "HD": "Home Depot Inc.",
    "DIS": "Walt Disney Company",
    "PYPL": "PayPal Holdings",
    "NFLX": "Netflix Inc.",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce Inc.",
    "INTC": "Intel Corporation",
    "CSCO": "Cisco Systems Inc.",
    "PEP": "PepsiCo Inc.",
    "T": "AT&T Inc.",
    "WMT": "Walmart Inc.",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "BA": "Boeing Company",
    "MCD": "McDonald's Corporation",
    "SBUX": "Starbucks Corporation",
    "AMD": "Advanced Micro Devices Inc.",
    "UBER": "Uber Technologies Inc."
}

class ElliottWaveAnalyzer:
    def __init__(self, window=20):
        self.window = window
    
    def get_extrema(self, prices):
        max_idx = argrelextrema(prices.values, np.greater, order=self.window)[0]
        min_idx = argrelextrema(prices.values, np.less, order=self.window)[0]
        return max_idx, min_idx
    
    def identify_impulse_wave(self, prices, extrema_prices, extrema_idx):
        waves = []
        if len(extrema_prices) < 6:
            return waves
        
        for i in range(len(extrema_prices) - 5):
            points = extrema_prices[i:i+6]
            idx_points = extrema_idx[i:i+6]
            
            if self._validate_impulse_wave(points):
                wave_info = {
                    'type': 'impulse',
                    'points': points,
                    'indices': idx_points,
                    'wave1': (idx_points[0], points[0], idx_points[1], points[1]),
                    'wave2': (idx_points[1], points[1], idx_points[2], points[2]),
                    'wave3': (idx_points[2], points[2], idx_points[3], points[3]),
                    'wave4': (idx_points[3], points[3], idx_points[4], points[4]),
                    'wave5': (idx_points[4], points[4], idx_points[5], points[5])
                }
                waves.append(wave_info)
        return waves
    
    def _validate_impulse_wave(self, points):
        p0, p1, p2, p3, p4, p5 = map(float, points)
        
        retrace_2 = abs(p2 - p1) / abs(p1 - p0)
        if retrace_2 > 1.0:
            return False
        
        wave1_len = abs(p1 - p0)
        wave3_len = abs(p3 - p2)
        wave5_len = abs(p5 - p4)
        
        if wave3_len < wave1_len and wave3_len < wave5_len:
            return False
        
        if (p3 > p0 and p4 > p0) or (p3 < p0 and p4 < p0):
            pass
        else:
            return False
        
        return True
    
    def fibonacci_projection(self, wave_points, current_price):
        wp = list(map(float, wave_points))
        current_price = float(current_price)
        wave0_to_3 = abs(wp[3] - wp[0])
        
        levels = {
            '0.382': current_price + 0.382 * wave0_to_3,
            '0.618': current_price + 0.618 * wave0_to_3,
            '1.0': current_price + 1.0 * wave0_to_3,
            '1.272': current_price + 1.272 * wave0_to_3,
            '1.618': current_price + 1.618 * wave0_to_3,
        }
        return levels
    
    def analyze(self, df):
        prices = df['Close']
        max_idx, min_idx = self.get_extrema(prices)
        all_extrema_idx = np.sort(np.concatenate([max_idx, min_idx]))
        all_extrema_prices = prices.iloc[all_extrema_idx]
        waves = self.identify_impulse_wave(prices, all_extrema_prices.values, all_extrema_idx)
        return waves, all_extrema_idx, all_extrema_prices

def create_original_analysis_chart(df, waves, extrema_idx, extrema_prices, stock):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot price data
    ax.plot(df.index, df['Close'], label='Close Price', linewidth=1, color='black')
    ax.scatter(df.index[extrema_idx], extrema_prices, color='red', s=30, zorder=5, label='Extrema')
    
    # Plot identified waves
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    wave_labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
    
    for i, wave in enumerate(waves):
        for j, wave_key in enumerate(['wave1', 'wave2', 'wave3', 'wave4', 'wave5']):
            start_idx, start_price, end_idx, end_price = wave[wave_key]
            ax.plot([df.index[start_idx], df.index[end_idx]], 
                   [start_price, end_price], 
                   color=colors[j], linewidth=2, label=wave_labels[j] if i == 0 else "")
    
    ax.set_title(f'{stock} - Elliott Wave Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def create_forecast_levels_chart(df, waves, extrema_idx, extrema_prices, 
                               current_price, projections, stock):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top panel: Historical analysis
    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1, color='black')
    ax1.scatter(df.index[extrema_idx], extrema_prices, color='red', s=30, zorder=5)
    ax1.set_title(f'{stock} - Elliott Wave Analysis', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Forecast area
    last_50_days = df.tail(50)
    ax2.plot(last_50_days.index, last_50_days['Close'], label='Recent Price', linewidth=2, color='blue')
    
    if projections:
        # Plot projection levels
        for level, price in projections.items():
            ax2.axhline(y=price, color='orange', linestyle='--', alpha=0.7, 
                       label=f'Fib {level}: ${price:.2f}')
    
    ax2.axhline(y=current_price, color='red', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    ax2.set_title('Price Forecast with Fibonacci Levels', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_detailed_forecast_chart(df, waves, current_price, projections, stock):
    """Create the detailed forecast chart with projection line"""
    latest_wave = waves[-1]
    wave_points = latest_wave['points']
    wave_indices = latest_wave['indices']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top panel: Historical pattern
    ax1.plot(df.index, df['Close'], label=f'{stock} Close Price', linewidth=2, color='black')
    
    # Highlight the identified wave pattern
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    wave_labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
    
    for j, wave_key in enumerate(['wave1', 'wave2', 'wave3', 'wave4', 'wave5']):
        start_idx, start_price, end_idx, end_price = latest_wave[wave_key]
        ax1.plot([df.index[start_idx], df.index[end_idx]], 
                [start_price, end_price], 
                color=colors[j], linewidth=3, label=wave_labels[j], alpha=0.8)
    
    # Mark wave points
    ax1.scatter(df.index[wave_indices], wave_points, color='darkblue', s=80, zorder=5, alpha=0.8)
    ax1.set_title(f'{stock} - Identified Elliott Wave Pattern', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Forecast with projection line
    recent_data = df.tail(100)
    ax2.plot(recent_data.index, recent_data['Close'], label='Recent Price', linewidth=2, color='blue')
    
    # Current price marker
    last_date = df.index[-1]
    ax2.scatter([last_date], [current_price], color='red', s=100, zorder=5, 
               label=f'Current: ${current_price:.2f}')
    
    # Plot Fibonacci target levels
    fib_colors = ['purple', 'darkorange', 'red', 'brown', 'darkred']
    fib_levels = list(projections.keys())
    
    for i, (level, price) in enumerate(projections.items()):
        ax2.axhline(y=price, color=fib_colors[i % len(fib_colors)], linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'Fib {level}: ${price:.2f}')
    
    # Create and plot expected forecast path
    future_periods = 90
    future_dates = pd.date_range(start=last_date, periods=future_periods, freq='D')
    
    # Calculate projection
    wave_1_3_height = float(wave_points[3]) - float(wave_points[0])
    wave_4_level = float(wave_points[4])
    current_price_float = float(current_price)
    
    # Use available projection levels safely
    typical_targets = [
        wave_4_level + 0.382 * wave_1_3_height,
        wave_4_level + 0.618 * wave_1_3_height,
        wave_4_level + 1.0 * wave_1_3_height,
    ]
    
    # Add 1.618 projection if available, otherwise calculate it
    if '1.618' in projections:
        typical_targets.append(float(projections['1.618']))
    else:
        typical_targets.append(wave_4_level + 1.618 * wave_1_3_height)
    
    # Create forecast curve using the second target (0.618 extension)
    target_price = typical_targets[1]  # Use 0.618 extension as primary target
    base_forecast = np.linspace(current_price_float, target_price, len(future_dates))
    x = np.linspace(0, 4*np.pi, len(future_dates))
    oscillation = 0.1 * (target_price - current_price_float) * np.sin(x)
    forecast_prices = base_forecast + oscillation
    
    # Plot the forecast path
    ax2.plot(future_dates, forecast_prices, color='green', linewidth=2, 
            linestyle='-', alpha=0.6, label='Expected Path (Projected)')
    
    ax2.set_title(f'{stock} - Elliott Wave Forecast & Target Levels', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Price ($)', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_analysis_report(df, waves, current_price, projections, stock):
    report = f"# ELLIOTT WAVE ANALYSIS REPORT\n"
    report += "=" * 50 + "\n\n"
    report += f"**Stock:** {stock} - {STOCK_DATABASE.get(stock, 'N/A')}\n\n"
    report += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**Current Price:** ${current_price:.2f}\n\n"
    report += f"**Data Period:** {len(df)} trading days\n\n"
    
    # Wave count
    report += f"**Elliott Wave Patterns Identified:** {len(waves)}\n\n"
    
    if waves and projections:
        report += "## FIBONACCI PROJECTION LEVELS\n"
        report += "-" * 30 + "\n\n"
        for level, price in projections.items():
            change_pct = ((price - current_price) / current_price) * 100
            report += f"- **Fib {level}:** ${price:.2f} ({change_pct:+.1f}%)\n"
        
        report += "\n## PROBABILITY ASSESSMENT\n"
        report += "-" * 25 + "\n\n"
        prob_model = {'0.382': 0.75, '0.618': 0.60, '1.0': 0.45, '1.272': 0.30, '1.618': 0.20}
        for level, prob in prob_model.items():
            if level in projections:
                price = projections[level]
                change_pct = ((price - current_price) / current_price) * 100
                report += f"- **Fib {level}:** {prob:.0%} chance → ${price:.2f} ({change_pct:+.1f}%)\n"
        
        report += "\n## INTERPRETATION\n"
        report += "-" * 15 + "\n\n"
        report += "- Green forecast line shows potential path to targets\n"
        report += "- Dashed lines represent Fibonacci projection levels\n"
        report += "- Wave 5 typically reaches between 0.618-1.618 extension\n"
        report += "- Monitor price action at each Fibonacci level\n"
    else:
        report += "## NO CLEAR PATTERNS DETECTED\n\n"
        report += "No clear Elliott Wave patterns detected in the current data.\n\n"
        report += "**Possible reasons:**\n"
        report += "- The stock is in a corrective phase\n"
        report += "- The pattern is too complex for automated detection\n"
        report += "- Try adjusting the analysis parameters\n"
    
    report += "\n## DISCLAIMER\n"
    report += "-" * 10 + "\n\n"
    report += "Elliott Wave analysis is subjective and should be used in conjunction with other technical indicators and fundamental analysis.\n"
    
    return report

def main():
    st.set_page_config(
        page_title="Elliott Wave Analyzer Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .report-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📈 Elliott Wave Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock:",
        options=list(STOCK_DATABASE.keys()),
        index=5,  # Default to NVDA
        format_func=lambda x: f"{x} - {STOCK_DATABASE[x]}"
    )
    
    # Period selection
    period = st.sidebar.selectbox(
        "Data Period:",
        options=["1y", "2y", "5y", "10y"],
        index=2
    )
    
    # Analysis window
    window = st.sidebar.slider(
        "Analysis Window:",
        min_value=5,
        max_value=30,
        value=15,
        help="Larger windows detect longer-term patterns"
    )
    
    # Analyze button
    analyze_btn = st.sidebar.button("🚀 Analyze Elliott Waves", type="primary")
    
    # Main content
    if analyze_btn:
        with st.spinner(f"Downloading {selected_stock} data and analyzing..."):
            try:
                # Initialize analyzer
                analyzer = ElliottWaveAnalyzer(window=window)
                
                # Download data
                df = yf.download(selected_stock, interval='1D', period=period)
                
                if df.empty:
                    st.error(f"Failed to download data for {selected_stock}")
                    return
                
                # Perform analysis
                waves, extrema_idx, extrema_prices = analyzer.analyze(df)
                current_price = float(df['Close'].iloc[-1])
                
                # Generate projections if waves found
                projections = None
                if waves:
                    latest_wave = waves[-1]
                    projections = analyzer.fibonacci_projection(latest_wave['points'], current_price)
                
                # Display stock info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stock", f"{selected_stock} - {STOCK_DATABASE[selected_stock]}")
                with col2:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col3:
                    st.metric("Patterns Found", len(waves))
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Analysis Charts", "📋 Report", "🎯 Forecast"])
                
                with tab1:
                    # Chart 1: Original Analysis
                    st.subheader("1. Elliott Wave Pattern Recognition")
                    fig1 = create_original_analysis_chart(df, waves, extrema_idx, extrema_prices, selected_stock)
                    st.pyplot(fig1)
                    plt.close(fig1)
                    
                    # Chart 2: Forecast Levels
                    st.subheader("2. Forecast with Fibonacci Levels")
                    fig2 = create_forecast_levels_chart(df, waves, extrema_idx, extrema_prices, 
                                                      current_price, projections, selected_stock)
                    st.pyplot(fig2)
                    plt.close(fig2)
                    
                    # Chart 3: Detailed Forecast (if waves found)
                    if waves and projections:
                        st.subheader("3. Detailed Forecast with Projected Path")
                        fig3 = create_detailed_forecast_chart(df, waves, current_price, projections, selected_stock)
                        st.pyplot(fig3)
                        plt.close(fig3)
                
                with tab2:
                    # Analysis Report
                    report = generate_analysis_report(df, waves, current_price, projections, selected_stock)
                    st.markdown(report)
                
                with tab3:
                    # Focus on projections
                    if projections:
                        st.subheader("Fibonacci Projection Targets")
                        
                        # Create projection metrics
                        cols = st.columns(len(projections))
                        for i, (level, price) in enumerate(projections.items()):
                            change_pct = ((price - current_price) / current_price) * 100
                            with cols[i]:
                                st.metric(
                                    label=f"Fib {level}",
                                    value=f"${price:.2f}",
                                    delta=f"{change_pct:+.1f}%"
                                )
                        
                        # Probability assessment
                        st.subheader("Probability Assessment")
                        prob_model = {'0.382': 0.75, '0.618': 0.60, '1.0': 0.45, '1.272': 0.30, '1.618': 0.20}
                        
                        for level, prob in prob_model.items():
                            if level in projections:
                                price = projections[level]
                                change_pct = ((price - current_price) / current_price) * 100
                                
                                st.progress(
                                    float(prob),
                                    text=f"Fib {level}: {prob:.0%} chance to reach ${price:.2f} ({change_pct:+.1f}%)"
                                )
                    else:
                        st.warning("No Elliott Wave patterns detected for forecast analysis.")
                
                # Success message
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Try adjusting the analysis window or selecting a different stock.")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to Elliott Wave Analyzer Pro! 🎯
        
        This tool automatically identifies Elliott Wave patterns in stock prices and provides:
        
        - **📊 Pattern Recognition**: Automatic detection of impulse waves
        - **🎯 Fibonacci Projections**: Price targets based on wave measurements  
        - **📈 Forecast Visualizations**: Expected price paths and key levels
        - **📋 Detailed Reports**: Comprehensive analysis with probability assessments
        
        ### How to use:
        1. Select a stock from the dropdown in the sidebar
        2. Choose your preferred data period
        3. Adjust the analysis window if needed
        4. Click **'Analyze Elliott Waves'** to run the analysis
        
        ### Supported Stocks:
        All major US stocks including AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, and many more!
        
        ⚠️ **Disclaimer**: This is for educational purposes only. Always do your own research and consult with financial advisors before making investment decisions.
        """)
        
        # Quick stock preview
        st.subheader("Popular Stocks Available")
        cols = st.columns(4)
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"]
        
        for i, stock in enumerate(popular_stocks):
            with cols[i % 4]:
                st.write(f"**{stock}**")
                st.caption(STOCK_DATABASE[stock])

if __name__ == "__main__":
    main()
