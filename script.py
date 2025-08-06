from itertools import combinations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.api import add_constant, OLS
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

from datapoint import get_klines

def fetch_data(token):
    # Get the kline data
    df = get_klines(token=token,
                    kline_type='15m',
                    start=datetime.now()- timedelta(days=180))
    
    # Focus on close prices
    close = df['c']
    close.name = token
    
    # Clean the data
    close = close[~close.index.duplicated(keep='last')].sort_index()
    close = close.dropna()
    
    return close

def optimize_hedge_ratio(data):
    if data.empty:
        return None
        
    x_name, y_name = data.columns
    
    # Check for zero or constant values
    if data[x_name].std() == 0 or data[y_name].std() == 0:
        return None
    
    # Prepare the data
    X = add_constant(data[x_name])
    y = data[y_name]
    
    try:
        # Use linear regression to compute optimum hedge ratio
        h_ratio = OLS(y, X).fit().params.iloc[1]
        
        if h_ratio <= 0 or np.isnan(h_ratio):
            return None
            
        return h_ratio
    except Exception as e:
        print(f"Error in optimize_hedge_ratio: {e}")
        return None

def evaluate_spread(data, h_ratio):
    x_name, y_name = data.columns
    
    # Compute the spread and size
    spread = data[y_name] - h_ratio * data[x_name]
    size = data[y_name] + h_ratio * data[x_name]
    spread.name = f'{y_name} - {h_ratio:.3g}{x_name}'
    
    # Evaluate the spread using ADF test
    p_value, t_stat = adfuller(spread)[:2]
    
    return {'p_value': p_value, 't_stat': t_stat, 'spread': spread, 'size': size}

def visualize_spread(spread, mean, std):
    plt.figure(figsize=(14, 7))
    plt.plot(spread.index, spread)
    plt.axhline(mean, color='r', linestyle='--', label='Mean')
    plt.axhline(mean + 2*std, color='g', linestyle='--', label='+2 Std')
    plt.axhline(mean - 2*std, color='g', linestyle='--')
    plt.axhline(mean + 3*std, color='b', linestyle='--', label='+3 Std')
    plt.axhline(mean - 3*std, color='b', linestyle='--')
    plt.title(spread.name)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
    
def simulate_trading(spread, size, mean, std, initial_balance=10_000, leverage=10):
    """
    Simulate trading based on spread, size, mean and standard deviation
    
    Parameters:
    -----------
    spread : array-like
        The spread values to trade on
    size : array-like
        The size values for position sizing
    mean : float
        Mean of the spread during training period
    std : float
        Standard deviation of the spread
    initial_balance : float, optional
        Starting balance, defaults to 10,000
    leverage : int, optional
        Trading leverage, defaults to 10
        
    Returns:
    --------
    pd.Series
        Series of account balance values
    """
    # Compute z-score
    z = (spread - mean) / std
    
    # Generate trading signals
    enter_long = np.where(np.diff(z <= -2) == 1, 1, np.nan)
    enter_short = np.where(np.diff(z >= 2), -1, np.nan)
    sl = np.where(np.diff(np.abs(z) >= 3) == 1, 0, np.nan)
    tp = np.where(np.diff(np.sign(z)) != 0, 0, np.nan)
    
    # Initial position
    initial_position = 0
    if 2 <= z[0] < 3: initial_position = -1
    if -3 < z[0] <= -2: initial_position = 1
    
    # Compute positions
    position = np.full(len(spread) - 1, np.nan)
    for signal in [enter_long, enter_short, sl, tp]:
        position = np.where(~np.isnan(position), position, signal)
    position = [initial_position] + list(position)
    position = pd.Series(position, index=spread.index).ffill()
    
    # Compute PnL
    delta_spread = np.diff(spread) / size[:-1]
    delta_balance = delta_spread * leverage * position.values[:-1]
    interest = np.where(delta_balance > -1, delta_balance + 1, 0)
    interest = [1] + list(interest)
    
    # Compute balance
    balance = initial_balance * np.cumprod(interest)
    return pd.Series(balance, index=spread.index)

def main():
    tokens = ['BTC', 'DOGE', 'SHIB', 'LINK', 'UNI', 'AAVE']
    
    # Fetch data for each token
    print("Fetching data...")
    df = pd.concat([fetch_data(token) for token in tokens],
                   axis=1, join='inner')
    
    if df.empty:
        print("Error: No data available for analysis")
        return
        
    print(f"Data shape: {df.shape}")
    
    # Split data into training and testing sets
    training_cutoff = int(0.8 * len(df))
    if training_cutoff == 0:
        print("Error: Not enough data points for analysis")
        return
        
    training_df = df.iloc[:training_cutoff]
    testing_df = df.iloc[training_cutoff:]
    
    print("Calculating hedge ratios...")
    # Calculate optimum hedge_ratios using training data
    h_ratios = {}
    for pair in combinations(tokens, 2):
        try:
            ratio = optimize_hedge_ratio(training_df[list(pair)])
            if ratio is not None:
                h_ratios[pair] = ratio
        except Exception as e:
            print(f"Warning: Could not calculate hedge ratio for {pair}: {e}")
            continue
    
    if not h_ratios:
        print("Error: No valid hedge ratios found")
        return
        
    print(f"Found {len(h_ratios)} valid hedge ratios")
    
    # Compute spreads for the duration of the testing data
    results = {pair: evaluate_spread(testing_df[list(pair)], h_ratio)
               for pair, h_ratio in h_ratios.items()}
    
    # Discard results with high p-value, if possible
    restrict_p = {pair: result for pair, result in results.items()
                  if result['p_value']<=0.05}
    if len(restrict_p) > 0: results = restrict_p
    
    # Use t-stat to select the best tradable spread
    top_pair = max(results.keys(),
                   key = lambda pair: results[pair]['t_stat'])
    hedge_ratio = h_ratios[top_pair]
    
    # Find the top spread during the testing and training periods for visualization
    top_spread_testing = results[top_pair]['spread']
    top_spread_testing_size = results[top_pair]['size']
    top_spread_training = evaluate_spread(training_df[list(top_pair)],
                                          hedge_ratio)['spread']
    
    # Use the training data, representing known data, to compute mean and std for visualization and trading simulation
    mean = top_spread_training.mean()
    std = top_spread_testing.std()
    
    # Visualize top spread for training and testing periods
    visualize_spread(top_spread_training, mean, std)
    visualize_spread(top_spread_testing, mean, std)
    
    simulation = pd.DataFrame(index = testing_df.index)
    
    # Compute trading signals during the simulation
    spread = top_spread_testing.values
    size = top_spread_testing_size.values
    
    # Compute positions for the trading bot    
    initial_position = 0
    if 2 <= z[0] < 3: initial_position = -1 # type: ignore
    if -3 < z[0] <= -2: initial_position = 1 # type: ignore
    
    position = np.full(len(spread) - 1, np.nan)
    for signal in [enter_long, enter_short, sl, tp]: # type: ignore
        position = np.where(~np.isnan(position), position, signal)
    position = [initial_position] + list(position)
    position = pd.Series(position, index=simulation.index).ffill()
    
    # Compute balance for the trading bot
    initial_balance = 10_000
    leverage = 10
    
    delta_spread = np.diff(spread) / size[:-1]
    delta_balance = delta_spread * leverage * position.values[:-1]
    interest = np.where(delta_balance > -1, delta_balance + 1, 0)
    interest = [1] + list(interest)
    
    balance = initial_balance * np.cumprod(interest)
    simulation['balance'] = balance
    
    # Compute balance for bitcoin HODLer
    bitcoin = testing_df['BTC'].values
    delta_bitcoin = np.diff(bitcoin) / bitcoin[:-1]
    delta_balance = delta_bitcoin * leverage
    interest = np.where(delta_balance > -1, delta_balance + 1, 0)
    interest = [1] + list(interest)
    
    balance = initial_balance * np.cumprod(interest)
    simulation['HODLer balance'] = balance
    
    # Plot balances of each strategy in the testing period
    plt.figure(figsize=(14, 7))
    simulation.plot()
    plt.title('Trading Simulation')
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
    
if __name__ == '__main__':
    main()