## EURUSD Backtesting Starter

This project gives you a minimal Python scaffold for backtesting EURUSD
strategies using OHLC data. It includes:

- a sample dataset (`data/example_eurusd.csv`) so you can run the script end to end
- a moving-average crossover strategy implementation
- a single-asset backtester that handles position tracking, transaction costs,
  and headline performance metrics

### Quick start

```bash
pip install -r requirements.txt
python backtest.py --csv data/example_eurusd.csv --fast 3 --slow 6
```

Use your own data by pointing `--csv` to a file that has at least these columns:
`time, open, high, low, close`. The timestamps can be hourly, daily, or any
regular frequency.

### Extending the scaffold

- Drop in your strategy logic by subclassing `Strategy` and overriding
  `generate_signals`.
- Add risk management by modifying the `Backtester._simulate` method.
- Export detailed results (equity curve, drawdowns, positions) from
  `BacktestResult.performance` for further analysis or plotting.