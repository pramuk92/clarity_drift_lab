import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

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

selected_stock = st.selectbox("Choose a stock:", list(stock_dict.keys()))
ticker = stock_dict[selected_stock]
