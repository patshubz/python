from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.api import add_constant, OLS
from statsmodels.tsa.stattools import adfuller
import pytest
from script import fetch_data, optimize_hedge_ratio, evaluate_spread, simulate_trading, visualize_spread


def make_dummy_series(n=100):
    idx = pd.date_range(end=datetime.now(), periods=n, freq='15min')  # Changed from '15T' to '15min'
    vals = np.linspace(1, 2, n) + np.random.normal(0, 0.001, n)
    return pd.Series(vals, index=idx)

@pytest.fixture
def dummy_df():
    a = make_dummy_series()
    b = 1.5*a + np.random.normal(0,0.001,len(a))
    return pd.concat([a.rename('A'), pd.Series(b, index=a.index, name='B')], axis=1)

def test_optimize_hedge_ratio_positive(dummy_df: pd.DataFrame):
    h = optimize_hedge_ratio(dummy_df)
    assert pytest.approx(h, rel=1e-2) == 1.5

def test_optimize_hedge_ratio_none(dummy_df: pd.DataFrame):
    dummy_df['B'] = dummy_df['A'] * -1  # Make it negative
    h = optimize_hedge_ratio(dummy_df)
    assert h is None

def test_optimize_hedge_ratio_empty():
    empty_df = pd.DataFrame(columns=['A', 'B'])
    h = optimize_hedge_ratio(empty_df)
    assert h is None

def test_fetch_data_no_duplicates(monkeypatch: pytest.MonkeyPatch):
    raw = pd.DataFrame({'c':[1,2,2,3]}, index=['2025-01-01']*4)
    def fake_get_klines(**kwargs): return raw
    monkeypatch.setattr('script.get_klines', fake_get_klines)
    s = fetch_data('X')
    assert not s.index.duplicated().any()

def test_fetch_data_sorted(monkeypatch: pytest.MonkeyPatch):
    raw = pd.DataFrame({'c':[3,1]}, index=['2025-01-02','2025-01-01'])
    monkeypatch.setattr('script.get_klines', lambda **kw: raw)
    s = fetch_data('X')
    assert list(s.index)==sorted(s.index)

def test_max_function_with_empty_sequence():
    import script
    try:
        result = max([], key=lambda p: p[1])
    except ValueError:
        result = None
    assert result is None

def test_evaluate_spread(dummy_df: pd.DataFrame):
    h_ratio = optimize_hedge_ratio(dummy_df)
    result = evaluate_spread(dummy_df, h_ratio)
    
    assert isinstance(result, dict)
    assert 'spread' in result
    assert 'size' in result
    assert isinstance(result['spread'], pd.Series)
    assert isinstance(result['size'], pd.Series)
    assert result['spread'].name == f'B - {h_ratio:.3g}A'
    
    # Check if the spread is stationary using ADF test
    adf_result = adfuller(result['spread'].dropna())
    assert adf_result[1] < 0.05  # p-value should be less than 0.05 for stationarity

def test_simulate_trading_basic(dummy_df):
    """Test basic trading simulation functionality"""
    h_ratio = optimize_hedge_ratio(dummy_df)
    result = evaluate_spread(dummy_df, h_ratio)
    
    mean = result['spread'].mean()
    std = result['spread'].std()
    
    balance = simulate_trading(
        result['spread'], 
        result['size'], 
        mean, 
        std, 
        initial_balance=10000,
        leverage=1
    )
    
    assert isinstance(balance, pd.Series)
    assert len(balance) == len(dummy_df)
    assert balance.iloc[0] == 10000  # Initial balance

def test_simulate_trading_leverage(dummy_df):
    """Test that leverage affects returns proportionally"""
    h_ratio = optimize_hedge_ratio(dummy_df)
    result = evaluate_spread(dummy_df, h_ratio)
    
    mean = result['spread'].mean()
    std = result['spread'].std()
    
    balance1 = simulate_trading(result['spread'], result['size'], mean, std, leverage=1)
    balance2 = simulate_trading(result['spread'], result['size'], mean, std, leverage=2)
    
    # Check if profits/losses scale with leverage
    returns1 = balance1.iloc[-1] / balance1.iloc[0] - 1
    returns2 = balance2.iloc[-1] / balance2.iloc[0] - 1
    assert abs(returns2) > abs(returns1)

def test_evaluate_spread_edge_cases():
    """Test evaluate_spread with edge cases"""
    # Create dummy data with known properties
    idx = pd.date_range(end=datetime.now(), periods=100, freq='15min')
    df = pd.DataFrame({
        'A': np.ones(100),  # Constant series
        'B': np.linspace(1, 2, 100)  # Linear trend
    }, index=idx)
    
    result = evaluate_spread(df, h_ratio=1.0)
    assert 'p_value' in result
    assert 't_stat' in result
    assert isinstance(result['spread'], pd.Series)
    assert isinstance(result['size'], pd.Series)

def test_visualize_spread_no_plot_shown(monkeypatch):
    """Test visualize_spread without showing the plot"""
    # Mock plt.show to prevent display
    shown = False
    def mock_show():
        nonlocal shown
        shown = True
    monkeypatch.setattr(plt, 'show', mock_show)
    
    spread = pd.Series(np.random.randn(100))
    mean = spread.mean()
    std = spread.std()
    
    visualize_spread(spread, mean, std)
    assert shown

@pytest.mark.parametrize("token", ['BTC', 'DOGE', 'INVALID_TOKEN'])
def test_fetch_data_different_tokens(token):
    """Test fetch_data with different tokens"""
    series = fetch_data(token)
    assert isinstance(series, pd.Series)
    assert not series.empty
    assert series.name == token

def test_optimize_hedge_ratio_constant_data():
    """Test optimize_hedge_ratio with constant data"""
    df = pd.DataFrame({
        'A': [1, 1, 1],
        'B': [2, 2, 2]
    })
    result = optimize_hedge_ratio(df)
    assert result is None

def test_optimize_hedge_ratio_perfect_correlation():
    """Test optimize_hedge_ratio with perfectly correlated data"""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [2, 4, 6]  # Perfect 2:1 ratio
    })
    result = optimize_hedge_ratio(df)
    assert pytest.approx(result, rel=1e-10) == 2.0
