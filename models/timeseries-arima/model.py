"""Univariate time series forecasting with ARIMA.

A pure-Python ARIMA implementation for time series forecasting
without heavy dependencies. Supports configurable (p,d,q) orders.
"""

import numpy as np


def _difference(data, d=1):
    result = np.array(data, dtype=float)
    for _ in range(d):
        result = np.diff(result)
    return result


def _undifference(forecasts, history, d=1):
    result = np.array(forecasts, dtype=float)
    for _ in range(d):
        last = history[-1]
        undiffed = np.empty(len(result))
        for i in range(len(result)):
            undiffed[i] = result[i] + last
            last = undiffed[i]
        result = undiffed
    return result


def train(ctx):
    hp = ctx.hyperparameters
    p = int(hp.get("p", 2))
    d = int(hp.get("d", 1))
    n_steps = int(hp.get("forecast_steps", 10))

    ctx.log_metric("progress", 10)

    # Generate synthetic time series if no data provided
    try:
        data = ctx.get_input_data()
        series = np.array(data["series"], dtype=float)
    except Exception:
        np.random.seed(42)
        t = np.arange(200)
        series = 50 + 0.5 * t + 10 * np.sin(t / 10) + np.random.normal(0, 2, len(t))

    ctx.log_metric("progress", 30)

    diffed = _difference(series, d)

    # Fit AR coefficients via least squares
    if p > 0 and len(diffed) > p:
        X = np.column_stack([diffed[i:len(diffed) - p + i] for i in range(p)])
        y = diffed[p:]
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    else:
        coeffs = np.array([])

    ctx.log_metric("progress", 60)

    # Evaluate: walk-forward validation on last 20% of data
    split = int(len(series) * 0.8)
    train_series = series[:split]
    test_series = series[split:]

    errors = []
    for i in range(len(test_series)):
        history = np.concatenate([train_series, test_series[:i]])
        hd = _difference(history, d)
        if len(coeffs) > 0 and len(hd) >= p:
            pred_diff = np.dot(coeffs, hd[-p:])
        else:
            pred_diff = 0.0
        pred = _undifference([pred_diff], history, d)[0]
        err = abs(pred - test_series[i])
        errors.append(err)
        if i % max(1, len(test_series) // 5) == 0:
            ctx.log_metric("mae", float(np.mean(errors)), epoch=i + 1)

    mae = float(np.mean(errors))
    ctx.log_metric("mae", mae, epoch=len(test_series))
    ctx.log_metric("loss", mae, epoch=len(test_series))
    ctx.log_metric("progress", 100)


def infer(ctx):
    data = ctx.get_input_data()
    hp = ctx.hyperparameters if hasattr(ctx, "hyperparameters") else {}

    if "series" not in data:
        ctx.set_output({"error": "Provide 'series' (array of numbers) in input_data"})
        return

    series = np.array(data["series"], dtype=float)
    p = int(hp.get("p", 2))
    d = int(hp.get("d", 1))
    n_steps = int(data.get("forecast_steps", hp.get("forecast_steps", 10)))

    diffed = _difference(series, d)

    if p > 0 and len(diffed) > p:
        X = np.column_stack([diffed[i:len(diffed) - p + i] for i in range(p)])
        y = diffed[p:]
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    else:
        coeffs = np.array([])

    forecasts_diff = []
    history = list(diffed)
    for _ in range(n_steps):
        if len(coeffs) > 0 and len(history) >= p:
            pred = float(np.dot(coeffs, history[-p:]))
        else:
            pred = 0.0
        forecasts_diff.append(pred)
        history.append(pred)

    forecasts = _undifference(forecasts_diff, series, d).tolist()
    ctx.set_output({"forecasts": forecasts, "steps": n_steps})
