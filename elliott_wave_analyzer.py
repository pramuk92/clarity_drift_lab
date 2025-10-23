import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings('ignore')

class ElliottWaveAnalyzer:
    def __init__(self, window=15):
        self.window = window

    def analyze(self, df):
        prices = df['Close'].values
        extrema_idx = argrelextrema(prices, np.greater, order=self.window)[0]
        extrema_idx = np.append(extrema_idx, argrelextrema(prices, np.less, order=self.window)[0])
        extrema_idx = np.sort(extrema_idx)

        extrema_prices = prices[extrema_idx]
        waves = []

        for i in range(len(extrema_idx) - 4):
            points = extrema_idx[i:i+5]
            wave = {'points': points}
            waves.append(wave)

        return waves, extrema_idx, extrema_prices

    def fibonacci_projection(self, wave_points, current_price):
        fib_levels = ['0.382', '0.618', '1.0', '1.272', '1.618']
        start = wave_points[0]
        end = wave_points[-1]
        wave_length = current_price - wave_points[-2]
        projections = {}

        for level in fib_levels:
            projections[level] = current_price + wave_length * float(level)

        return projections

def plot_elliott_forecast(df, waves, current_price, projections, future_periods=90):
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=1, color='black')

    for wave in waves:
        points = wave['points']
        plt.plot(df.index[points], df['Close'].iloc[points], marker='o', linestyle='-', label='Elliott Wave')

    future_dates = pd.date_range(df.index[-1], periods=future_periods, freq='D')
    for level, price in projections.items():
        plt.axhline(y=price, linestyle='--', label=f'Fib {level}: ${price:.2f}', alpha=0.6)

    plt.axhline(y=current_price, color='red', linestyle='-', linewidth=2, label=f'Current: ${current_price:.2f}')
    plt.title('Elliott Wave Forecast & Target Levels', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Price ($)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
