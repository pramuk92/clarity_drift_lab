import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elliott_wave_analyzer import ElliottWaveAnalyzer, plot_elliott_forecast

st.set_page_config(page_title="Elliott Wave Forecast", layout="wide")

st.title("📈 Elliott Wave Forecast Dashboard")

# Disclaimer block
st.markdown("""
<div style='background-color:#f9f9f9; padding:10px; border-left:5px solid #ff4b4b'>
<b>Disclaimer:</b><br>
This tool is for educational and diagnostic purposes only. It does not constitute financial advice or trading recommendations.<br><br>
<b>CFTC Disclaimer:</b><br>
Trading financial instruments involves risk and may not be suitable for all investors. Past performance is not indicative of future results. Always consult with a registered financial advisor before making trading decisions.
</div>
""", unsafe_allow_html=True)

# Dictionary of major stocks
stock_dict = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Tesla Inc. (TSLA)": "TSLA",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Meta Platforms Inc. (META)": "META",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "JPMorgan Chase & Co. (JPM)": "JPM",
    "Exxon Mobil Corp. (XOM)": "XOM",
    "Johnson & Johnson (JNJ)": "JNJ"
}

selected_stock = st.selectbox("Choose a stock:", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]

df = yf.download(ticker, interval='1d', period='5y')

if df.empty:
    st.error("Failed to download data.")
else:
    analyzer = ElliottWaveAnalyzer(window=15)
    waves, extrema_idx, extrema_prices = analyzer.analyze(df)
    current_price = float(df['Close'].iloc[-1])
    projections = analyzer.fibonacci_projection(waves[-1]['points'], current_price) if waves else None

    st.subheader(f"Forecast for {selected_stock}")
    if projections:
        fig = plot_elliott_forecast(df, waves, current_price, projections, future_periods=90)
        st.pyplot(fig)
    else:
        st.warning("No valid Elliott Wave patterns detected.")
