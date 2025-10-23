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
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'hspace': 0.4})

    # Subplot 1: Full historical price with wave overlays
    axs[0].plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1)
    for wave in waves:
        points = wave['points']
        axs[0].plot(df.index[points], df['Close'].iloc[points], marker='o', linestyle='-', label='Elliott Wave')
    axs[0].set_title('Historical Price with Elliott Waves')
    axs[0].set_ylabel('Price ($)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Subplot 2: Recent price action
    recent_df = df.tail(50)
    axs[1].plot(recent_df.index, recent_df['Close'], label='Recent Price', color='blue', linewidth=2)
    axs[1].set_title('Recent Price Action (Last 50 Days)')
    axs[1].set_ylabel('Price ($)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Subplot 3: Forecast levels
    axs[2].plot(recent_df.index, recent_df['Close'], label='Recent Price', color='blue', linewidth=2)
    for level, price in projections.items():
        axs[2].axhline(y=price, linestyle='--', color='orange', alpha=0.7, label=f'Fib {level}: ${price:.2f}')
    axs[2].axhline(y=current_price, color='red', linestyle='-', linewidth=2, label=f'Current: ${current_price:.2f}')

    # Dynamic y-axis scaling
    all_prices = list(projections.values()) + [current_price]
    price_range = max(all_prices) - min(all_prices)
    padding = price_range * 0.2  # Add 20% vertical padding
    axs[2].set_ylim(min(all_prices) - padding, max(all_prices) + padding)

    axs[2].set_title('Forecast with Fibonacci Levels')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Price ($)')
    axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
