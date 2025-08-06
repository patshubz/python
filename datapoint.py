# datapoint.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_klines(token, kline_type, start, end=None):
    # Generate synthetic data for testing
    end = end or datetime.now()
    idx = pd.date_range(start=start, end=end, freq='15min')
    
    # Create random walk data
    np.random.seed(42)  # For reproducibility
    price = 100 * np.exp(np.random.randn(len(idx)) * 0.02).cumsum()
    
    # Add some token-specific variation
    multiplier = {
        'BTC': 1000,
        'DOGE': 0.1,
        'SHIB': 0.00001,
        'LINK': 10,
        'UNI': 5,
        'AAVE': 50
    }.get(token, 1)
    
    data = pd.DataFrame({
        'c': price * multiplier
    }, index=idx)
    
    return data
