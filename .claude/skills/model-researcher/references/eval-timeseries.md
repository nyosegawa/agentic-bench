# Time Series Model Evaluation Guide

## Smoke Test
- Load model (e.g., TimesFM via `timesfm` package, or transformers)
- Prepare a simple synthetic time series: `np.sin(np.linspace(0, 4*np.pi, 100))`
- Run forecast and verify output shape matches expected horizon

## Quality Check — Test Data

Run on at least 2 datasets:

1. **Synthetic (sin wave)**: Predictable pattern to verify basic functionality
2. **Real-world**: Use a simple dataset (stock prices, temperature, energy consumption)

```python
import numpy as np

# Synthetic test data
t = np.linspace(0, 8 * np.pi, 200)
data = np.sin(t) + 0.1 * np.random.randn(len(t))

# Split: 80% train, 20% test
context = data[:160]
ground_truth = data[160:]

# Forecast
forecast = model.predict(context, horizon=40)
```

Visualize with matplotlib:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(range(160), context, label="Context")
plt.plot(range(160, 200), ground_truth, label="Ground Truth", linestyle="--")
plt.plot(range(160, 200), forecast, label="Forecast")
plt.legend()
plt.savefig("artifacts/forecast.png")
```

Evaluate:
- Does the forecast follow the trend?
- How does accuracy degrade with longer horizons?
- Are seasonal patterns captured?

## Performance Measurement

```python
import time

times = []
for _ in range(5):
    start = time.perf_counter()
    forecast = model.predict(context, horizon=40)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

median_time = sorted(times)[len(times) // 2]
```

Key metrics:
- **MAE** (Mean Absolute Error): `np.mean(np.abs(forecast - ground_truth))`
- **RMSE**: `np.sqrt(np.mean((forecast - ground_truth)**2))`
- **Inference time** per forecast
- **Supported horizons** (max forecast length)

## Common Issues
- Shape mismatch: Check if model expects (batch, seq_len) or (seq_len,)
- NaN outputs: Input data may need normalization
- CPU-only: Many time series models don't need GPU; verify before allocating one
- Dependencies: TimesFM needs specific JAX/PyTorch version
