# Stock Price Trend Prediction using Quantum Neural Networks (QNNs)

This project implements a hybrid deep learning model that combines attention mechanisms with a simulated quantum layer to predict short-term stock price trends using Limit Order Book (LOB) data.

It explores how Quantum Neural Networks (QNNs), even in simulated environments, can help capture complex, high-dimensional patterns in financial time series data and compares their performance with established benchmarks.

---

## Project Overview

- **Task**: Predict the future direction (Up, Down, or Stable) of the stock mid-price.
- **Input**: 144-dimensional LOB snapshot for a given time.
- **Output**: A 5-element vector representing predicted trends at horizons $k = \{1, 2, 3, 5, 10\}$.
- **Model**: PyTorch-based neural network with an attention block + simulated quantum layer (via a non-linear transformation).

---

## Dataset

We use the [FI-2010](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) dataset, a publicly available benchmark for mid-price forecasting based on LOB data.

>  The dataset is too large to be uploaded directly to GitHub.

You can download it from:

🔗 **https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649**


