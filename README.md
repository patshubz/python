Pairs Trading Bot
A Python-based pairs trading bot that implements statistical arbitrage strategies using cryptocurrency pairs.

Features
Fetches historical cryptocurrency price data
Calculates optimal hedge ratios using linear regression
Tests for cointegration using Augmented Dickey-Fuller test
Visualizes spread patterns and trading signals
Simulates trading performance with configurable leverage
Compares strategy performance against HODLing Bitcoin
Installation
Clone the repository:

git clone https://github.com/patshubz/python.git
cd python

Create and activate virtual environment:
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

Install dependencies:
pip install -r requirements.txt

Usage
Run the main script:

python script.py

The script will:

Fetch historical data for configured cryptocurrency pairs
Find the most statistically significant pair
Display spread visualization
Show trading simulation results
Configuration
Modify these parameters in script.py:

tokens: List of cryptocurrencies to analyze
initial_balance: Starting balance for simulation
leverage: Trading leverage multiplier
Trading thresholds in simulate_trading() function

Testing
Run tests with coverage report:

pytest --cov=. --cov-report=term --cov-report=html

Stucture

python/
├── script.py          # Main trading bot implementation
├── datapoint.py       # Data fetching functionality
├── test_solution.py   # Test suite
└── requirements.txt   # Project dependencies

Requirements
Python 3.9+
Dependencies listed in requirements.txt
License
MIT License

Contributing
Fork the repository
Create a feature branch
Commit changes
Push to the branch
Open a Pull Request

