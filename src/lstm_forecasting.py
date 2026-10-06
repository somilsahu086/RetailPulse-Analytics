"""
RetailPulse — Demand Forecasting Module
PyTorch LSTM Sequence Forecasting Framework

Implements a deterministic, time-series safe Long Short-Term Memory (LSTM)
neural network for product-level demand forecasting.

Key Architectural Safeguards:
1. Scaling without leakage: MinMax scaler is fitted strictly on historical
   training observations prior to sequence creation.
2. Sequence window generation: sliding lookback window (e.g. 30 or 60 days).
3. Recursive multi-step forecasting: out-of-sample projections update sequences
   step-by-step without accessing holdout ground truth.
4. Non-negative demand constraint: predictions are post-clipped at 0.0.
5. Deterministic reproducibility: sets PyTorch and NumPy random seeds.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MinMaxDemandScaler:
    """
    Lightweight, zero-safe MinMax scaler for univariate demand sequences.
    Fitted strictly on training data to prevent lookahead data leakage.
    """
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.min_range, self.max_range = feature_range
        self.data_min: float = 0.0
        self.data_max: float = 1.0
        self.fitted: bool = False

    def fit(self, data: Union[np.ndarray, pd.Series, list]) -> "MinMaxDemandScaler":
        arr = np.asarray(data, dtype=float)
        self.data_min = float(np.min(arr))
        self.data_max = float(np.max(arr))
        self.fitted = True
        return self

    def transform(self, data: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler must be fitted on training data before transform.")
        arr = np.asarray(data, dtype=float)
        spread = self.data_max - self.data_min
        if spread == 0.0:
            return np.zeros_like(arr)
        scaled = (arr - self.data_min) / spread
        return scaled * (self.max_range - self.min_range) + self.min_range

    def inverse_transform(self, scaled: Union[np.ndarray, list]) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform.")
        arr = np.asarray(scaled, dtype=float)
        spread = self.data_max - self.data_min
        if spread == 0.0:
            return np.full_like(arr, self.data_min)
        unscaled = (arr - self.min_range) / (self.max_range - self.min_range)
        return unscaled * spread + self.data_min


class DemandLSTM(nn.Module):
    """
    PyTorch LSTM architecture for demand sequence forecasting.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take hidden state from the last time step
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out


def create_sequences(
    series: np.ndarray,
    lookback: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D time-series into sliding window sequences:
    X[i] = [y_{i}, y_{i+1}, ..., y_{i+lookback-1}]
    y[i] = y_{i+lookback}

    Parameters
    ----------
    series : np.ndarray
        1D scaled demand array.
    lookback : int
        Window size in days.

    Returns
    -------
    X : np.ndarray of shape (N, lookback, 1)
    y : np.ndarray of shape (N, 1)
    """
    X_list, y_list = [], []
    for i in range(len(series) - lookback):
        X_list.append(series[i : i + lookback])
        y_list.append(series[i + lookback])
    if len(X_list) == 0:
        return np.empty((0, lookback, 1)), np.empty((0, 1))
    X = np.array(X_list).reshape(-1, lookback, 1)
    y = np.array(y_list).reshape(-1, 1)
    return X, y


def train_predict_lstm(
    history_df: pd.DataFrame,
    horizon: int = 30,
    lookback: int = 30,
    hidden_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 25,
    lr: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
    date_col: str = "Date",
    demand_col: str = "Demand"
) -> pd.DataFrame:
    """
    Train PyTorch LSTM on historical demand and generate 30-day recursive forecasts.

    Strict Anti-Leakage Safeguards:
    1. MinMax scaling parameters (data_min, data_max) are computed strictly
       from history_df[demand_col].
    2. Sequence generation operates exclusively on the training timeline.
    3. Multi-step forecasting rolls forward using predicted values, with no
       access to holdout actuals.

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical product time series.
    horizon : int
        Number of calendar days ahead to forecast.
    lookback : int
        Number of historical lookback days for sequence window.
    hidden_dim : int
        LSTM hidden units.
    num_layers : int
        Number of stacked LSTM layers.
    epochs : int
        Training iterations.
    lr : float
        Learning rate for Adam optimizer.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed for reproducibility.
    date_col : str
        Date column name.
    demand_col : str
        Target column name.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: ['Date', 'Predicted_Demand', 'Model']
    """
    if history_df.empty:
        raise ValueError("Cannot train LSTM on an empty history DataFrame.")

    # Set deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sort history chronologically
    df_sorted = history_df.sort_values(by=date_col).copy()
    raw_demand = df_sorted[demand_col].astype(float).values

    if len(raw_demand) <= lookback:
        raise ValueError(
            f"History length ({len(raw_demand)}) must be strictly greater than lookback ({lookback})."
        )

    # 1. Scale data strictly on history
    scaler = MinMaxDemandScaler()
    scaler.fit(raw_demand)
    scaled_demand = scaler.transform(raw_demand)

    # 2. Generate sliding sequences
    X_train, y_train = create_sequences(scaled_demand, lookback=lookback)

    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # 3. Model setup
    model = DemandLSTM(input_size=1, hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. Training loop
    model.train()
    dataset_size = len(X_tensor)
    effective_batch = min(batch_size, dataset_size)

    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, effective_batch):
            indices = permutation[i : i + effective_batch]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 5. Recursive Multi-Step Forecasting
    model.eval()
    last_date = pd.to_datetime(df_sorted[date_col].iloc[-1]).floor("D")
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Seed the rolling sequence with the final lookback window of scaled history
    current_seq = scaled_demand[-lookback:].copy().reshape(1, lookback, 1)

    predictions_scaled = []
    with torch.no_grad():
        for _ in range(horizon):
            seq_tensor = torch.tensor(current_seq, dtype=torch.float32)
            pred_step = model(seq_tensor).item()

            # Bound predictions within plausible scaled range
            pred_step = max(0.0, pred_step)
            predictions_scaled.append(pred_step)

            # Roll sequence forward by 1 step
            new_val = np.array([[[pred_step]]])
            current_seq = np.append(current_seq[:, 1:, :], new_val, axis=1)

    # 6. Inverse transform and enforce non-negative retail demand
    unscaled_preds = scaler.inverse_transform(predictions_scaled)
    final_preds = np.maximum(0.0, unscaled_preds)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Demand": final_preds,
        "Model": "LSTM"
    })
