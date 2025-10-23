import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')
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

st.title("📈 Ell-iad Wave Forecast Dashboard")

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

# Download data
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
        plot_elliott_forecast(df, waves, current_price, projections, future_periods=90)
    else:
        st.warning("No valid Elliott Wave patterns detected.")

class ElliottWaveAnalyzer:
    def __init__(self, window=20):
        self.window = window  # window for local extrema detection

    def get_extrema(self, prices):
        """Find local minima and maxima"""
        # Find local maxima
        max_idx = argrelextrema(prices.values, np.greater, order=self.window)[0]
        # Find local minima
        min_idx = argrelextrema(prices.values, np.less, order=self.window)[0]

        return max_idx, min_idx

    def identify_impulse_wave(self, prices, extrema_prices, extrema_idx):
        """Identify potential impulse waves (5-wave patterns)"""
        waves = []

        if len(extrema_prices) < 6:
            return waves

        for i in range(len(extrema_prices) - 5):
            points = extrema_prices[i:i+6]
            idx_points = extrema_idx[i:i+6]

            # Basic Elliott Wave rules for impulse waves
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
        """Validate Elliott Wave impulse pattern rules"""
        # Convert to float to avoid array issues
        p0, p1, p2, p3, p4, p5 = map(float, points)

        # Wave 2 should not retrace more than 100% of Wave 1
        retrace_2 = abs(p2 - p1) / abs(p1 - p0)
        if retrace_2 > 1.0:
            return False

        # Wave 3 should not be the shortest among waves 1, 3, and 5
        wave1_len = abs(p1 - p0)
        wave3_len = abs(p3 - p2)
        wave5_len = abs(p5 - p4)

        if wave3_len < wave1_len and wave3_len < wave5_len:
            return False

        # Wave 4 should not overlap with Wave 1
        if (p3 > p0 and p4 > p0) or (p3 < p0 and p4 < p0):
            pass  # Valid case
        else:
            return False

        return True

    def fibonacci_projection(self, wave_points, current_price):
        """Calculate Fibonacci projection levels"""
        # Convert all values to float to avoid numpy array issues
        wp = list(map(float, wave_points))
        current_price = float(current_price)

        wave0_to_3 = abs(wp[3] - wp[0])

        # Common Fibonacci targets
        levels = {
            '0.382': current_price + 0.382 * wave0_to_3,
            '0.618': current_price + 0.618 * wave0_to_3,
            '1.0': current_price + 1.0 * wave0_to_3,
            '1.272': current_price + 1.272 * wave0_to_3,
            '1.618': current_price + 1.618 * wave0_to_3,
        }

        return levels

    def analyze(self, df):
        """Main analysis function"""
        prices = df['Close']

        # Get extrema
        max_idx, min_idx = self.get_extrema(prices)

        # Combine and sort all extrema
        all_extrema_idx = np.sort(np.concatenate([max_idx, min_idx]))
        all_extrema_prices = prices.iloc[all_extrema_idx]

        # Identify waves
        waves = self.identify_impulse_wave(prices, all_extrema_prices.values, all_extrema_idx)

        return waves, all_extrema_idx, all_extrema_prices

def plot_elliott_waves(df, waves, extrema_idx, extrema_prices):
    """Plot the price data with identified Elliott Waves"""
    plt.figure(figsize=(15, 8))

    # Plot price data
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=1, color='black')

    # Plot extrema
    plt.scatter(df.index[extrema_idx], extrema_prices, color='red', s=30, zorder=5, label='Extrema')

    # Plot identified waves
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    wave_labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']

    for i, wave in enumerate(waves):
        for j, wave_key in enumerate(['wave1', 'wave2', 'wave3', 'wave4', 'wave5']):
            start_idx, start_price, end_idx, end_price = wave[wave_key]
            plt.plot([df.index[start_idx], df.index[end_idx]],
                    [start_price, end_price],
                    color=colors[j], linewidth=2, label=wave_labels[j] if i == 0 else "")

        # Mark wave start and end points
        wave_points = wave['points']
        wave_indices = wave['indices']
        plt.scatter(df.index[wave_indices], wave_points, color='blue', s=50, zorder=5)

    plt.title('Elliott Wave Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def forecast_levels(df, waves, current_price):
    """Generate Fibonacci-based forecasts"""
    if waves:
        latest_wave = waves[-1]  # Use most recent wave
        projections = analyzer.fibonacci_projection(latest_wave['points'], current_price)

        print("\n=== ELLIOTT WAVE FORECAST ===")
        print(f"Current Price: ${current_price:.2f}")
        print("\nFibonacci Projection Levels:")
        for level, price in projections.items():
            change_pct = ((price - current_price) / current_price) * 100
            print(f"{level}: ${price:.2f} ({change_pct:+.1f}%)")

        return projections
    else:
        print("No clear Elliott Wave patterns detected for forecasting.")
        return None

# Main execution
if __name__ == "__main__":
    # Download data
    print("Downloading data...")
    df = yf.download(ticker, interval=interval, period=period)

    if df.empty:
        print("Failed to download data.")
    else:
        # Initialize analyzer
        analyzer = ElliottWaveAnalyzer(window=15)

        # Perform analysis
        waves, extrema_idx, extrema_prices = analyzer.analyze(df)

        # Plot results
        plot_elliott_waves(df, waves, extrema_idx, extrema_prices)

        # Get current price
        current_price = float(df['Close'].iloc[-1])

        projections = forecast_levels(df, waves, current_price)

        # Print wave count
        print(f"\nIdentified {len(waves)} potential Elliott Wave patterns")

        # Additional analysis: Recent price action
        recent_change = ((current_price - float(df['Close'].iloc[-30])) / float(df['Close'].iloc[-30])) * 100
        print(f"\nRecent Performance (30 days): {recent_change:+.1f}%")

        # Plot with forecast levels
        if projections:
            plt.figure(figsize=(15, 10))

            # Historical data
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['Close'], label='Close Price', linewidth=1, color='black')
            plt.scatter(df.index[extrema_idx], extrema_prices, color='red', s=30, zorder=5)
            plt.title('Elliott Wave Analysis')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Forecast area
            plt.subplot(2, 1, 2)
            last_50_days = df.tail(50)
            plt.plot(last_50_days.index, last_50_days['Close'], label='Recent Price', linewidth=2, color='blue')

            # Plot projection levels
            for level, price in projections.items():
                plt.axhline(y=price, color='orange', linestyle='--', alpha=0.7,
                           label=f'Fib {level}: ${price:.2f}')

            plt.axhline(y=current_price, color='red', linestyle='-', linewidth=2, label=f'Current: ${current_price:.2f}')
            plt.title('Price Forecast with Fibonacci Levels')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

def plot_elliott_forecast(df, waves, current_price, projections, future_periods=60):
    """
    Plot Elliott Wave forecast with expected path and target levels
    """
    if not waves:
        print("No Elliott Wave patterns detected for forecasting.")
        return

    # Get the most recent wave pattern
    latest_wave = waves[-1]
    wave_points = latest_wave['points']
    wave_indices = latest_wave['indices']

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot 1: Complete historical data with wave annotation
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='black')

    # Highlight the identified wave pattern
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    wave_labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']

    for j, wave_key in enumerate(['wave1', 'wave2', 'wave3', 'wave4', 'wave5']):
        start_idx, start_price, end_idx, end_price = latest_wave[wave_key]
        plt.plot([df.index[start_idx], df.index[end_idx]],
                [start_price, end_price],
                color=colors[j], linewidth=3, label=wave_labels[j], alpha=0.8)

    # Mark wave points
    plt.scatter(df.index[wave_indices], wave_points, color='darkblue', s=80, zorder=5, alpha=0.8)

    plt.title('Identified Elliott Wave Pattern', fontsize=14, fontweight='bold')
    plt.ylabel('Price ($)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Forecast with expected path
    plt.subplot(2, 1, 2)

    # Get the last 100 days for context
    recent_data = df.tail(100)
    plt.plot(recent_data.index, recent_data['Close'], label='Recent Price', linewidth=2, color='blue')

    # Current price marker
    last_date = df.index[-1]
    plt.scatter([last_date], [current_price], color='red', s=100, zorder=5, label=f'Current: ${current_price:.2f}')

    # Create future dates for forecast
    future_dates = pd.date_range(start=last_date, periods=future_periods+1, freq='D')[1:]

    # Plot Fibonacci target levels - USE CONSISTENT KEYS
    fib_colors = ['purple', 'darkorange', 'red', 'brown', 'darkred']
    # Use the actual keys from projections to avoid KeyErrors
    fib_levels = list(projections.keys())

    for i, (level, price) in enumerate(projections.items()):
        plt.axhline(y=price, color=fib_colors[i % len(fib_colors)], linestyle='--', alpha=0.7,
                   linewidth=2, label=f'Fib {level}: ${price:.2f}')

    # Create and plot expected forecast path (simplified Elliott Wave projection)
    if len(wave_points) >= 5:
        # Calculate wave characteristics for projection
        wave_1_3_height = float(wave_points[3]) - float(wave_points[0])  # Wave 1-3 move
        wave_4_level = float(wave_points[4])
        current_price_float = float(current_price)

        # Simple projection: Assume Wave 5 typically reaches between 0.618 to 1.618 of Wave 1-3
        typical_targets = [
            wave_4_level + 0.382 * wave_1_3_height,
            wave_4_level + 0.618 * wave_1_3_height,
            wave_4_level + 1.0 * wave_1_3_height,
            float(projections['1.618'])  # Use the Fibonacci projection as maximum target
        ]

        # Create forecast dates including current date
        forecast_dates = pd.date_range(start=last_date, periods=future_periods, freq='D')

        # Create a smooth forecast curve (simplified)
        base_forecast = np.linspace(current_price_float, typical_targets[2], len(forecast_dates))

        # Add some Elliott Wave-like oscillation to the forecast
        x = np.linspace(0, 4*np.pi, len(forecast_dates))
        oscillation = 0.1 * (typical_targets[2] - current_price_float) * np.sin(x)
        forecast_prices = base_forecast + oscillation

        # Plot the forecast path
        plt.plot(forecast_dates, forecast_prices, color='green', linewidth=2,
                linestyle='-', alpha=0.6, label='Expected Path (Projected)')

        # Mark key forecast points - FIXED: Ensure we don't exceed array bounds
        key_points = [min(10, len(forecast_dates)-1),
                     min(30, len(forecast_dates)-1),
                     len(forecast_dates)-1]

        # Prepare data for scatter plot
        scatter_dates = []
        scatter_prices = []
        for point in key_points:
            if point < len(forecast_dates):
                scatter_dates.append(forecast_dates[point])
                scatter_prices.append(forecast_prices[point])

        # Plot scatter points only if we have data
        if scatter_dates and scatter_prices:
            plt.scatter(scatter_dates, scatter_prices, color='darkgreen', s=50, alpha=0.8)

    plt.title('Elliott Wave Forecast & Target Levels', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Price ($)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Add annotations for key levels
    max_projection = max(projections.values())
    min_projection = min(projections.values())

    plt.ylim(min(current_price, min_projection) * 0.95,
             max(current_price, max_projection) * 1.05)

    plt.tight_layout()
    plt.show()

    # Print detailed forecast analysis
    print("\n" + "="*60)
    print("ELLIOTT WAVE FORECAST ANALYSIS")
    print("="*60)
    print(f"Current Position: Wave 5 (projected)")
    print(f"Current Price: ${current_price:.2f}")
    print("\nKey Fibonacci Target Levels:")
    print("-" * 30)
    for level in fib_levels:
        price = projections[level]
        change_pct = ((price - current_price) / current_price) * 100
        print(f"Fib {level:>5}: ${price:8.2f} ({change_pct:+.1f}%)")

    print("\nInterpretation:")
    print("-" * 30)
    print("• Green line shows potential price path to targets")
    print("• Dashed lines represent Fibonacci projection levels")
    print("• Wave 5 typically reaches between 0.618-1.618 extension")
    print("• Monitor price action at each Fibonacci level for reversals")

# Run the forecast plot after your existing analysis
print("\nGenerating Elliott Wave Forecast...")
plot_elliott_forecast(df, waves, current_price, projections, future_periods=90)

# Additional: Probability analysis - FIXED KEY CONSISTENCY
def probability_analysis(projections, current_price):
    """Provide probability assessment of different targets"""
    print("\n" + "="*60)
    print("PROBABILITY ASSESSMENT")
    print("="*60)

    # Use the actual keys from projections to avoid KeyErrors
    fib_levels = list(projections.keys())

    # Simple probability model based on Fibonacci levels
    # Map the probability to the actual keys in projections
    prob_model = {
        '0.382': 0.75,   # High probability - minor extension
        '0.618': 0.60,   # Medium probability - common target
        '1.0': 0.45,     # Lower probability - equal move (NOTE: using '1.0' not '1.000')
        '1.272': 0.30,   # Low probability - extended
        '1.618': 0.20    # Very low probability - highly extended
    }

    print("Target Probability Estimates:")
    print("-" * 40)

    for level in fib_levels:
        # Use the probability from prob_model if available, otherwise use a default
        prob = prob_model.get(level, 0.25)  # Default to 25% if level not in prob_model
        price = projections[level]
        change_pct = ((price - current_price) / current_price) * 100
        print(f"Fib {level}: {prob:.0%} chance → ${price:.2f} ({change_pct:+.1f}%)")

    print("\nTrading Considerations:")
    print("-" * 40)
    print("• Higher probability targets (0.382, 0.618) often act as initial objectives")
    print("• Extended targets (1.272, 1.618) represent optimistic scenarios")
    print("• Always use proper risk management and confirm with price action")

# Run probability analysis
if projections:
    probability_analysis(projections, current_price)
