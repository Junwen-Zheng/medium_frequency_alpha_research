from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ModelResult:
    name: str
    predictions: pd.Series
    feature_importance: pd.Series | None = None


def _clean_model_frame(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    clip_value: float = 10.0,
    target_clip_value: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure sklearn models only receive finite numeric feature/target values.

    Feature clipping protects model fitting from public OHLCV data artifacts.
    Target clipping limits extreme forward-return artifacts such as bad adjusted
    prices or split/dividend issues. This is a defensive data-quality guard, not
    evidence of alpha.
    """
    required = features + [target]

    train_clean = train.copy()
    test_clean = test.copy()

    train_clean[required] = train_clean[required].replace([np.inf, -np.inf], np.nan)
    test_clean[required] = test_clean[required].replace([np.inf, -np.inf], np.nan)

    train_clean = train_clean.dropna(subset=required)
    test_clean = test_clean.dropna(subset=required)

    train_clean[features] = train_clean[features].astype(float).clip(-clip_value, clip_value)
    test_clean[features] = test_clean[features].astype(float).clip(-clip_value, clip_value)

    train_clean[target] = train_clean[target].astype(float).clip(-target_clip_value, target_clip_value)
    test_clean[target] = test_clean[target].astype(float).clip(-target_clip_value, target_clip_value)

    if train_clean.empty:
        raise ValueError("No finite training rows remain after model input cleaning.")
    if test_clean.empty:
        raise ValueError("No finite test rows remain after model input cleaning.")

    return train_clean, test_clean


def fit_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> ModelResult:
    train, test = _clean_model_frame(train, test, features, target)

    x_train = train[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)

    if not np.isfinite(x_train).all():
        raise ValueError("Non-finite values found in ridge x_train after cleaning.")
    if not np.isfinite(y_train).all():
        raise ValueError("Non-finite values found in ridge y_train after cleaning.")
    if not np.isfinite(x_test).all():
        raise ValueError("Non-finite values found in ridge x_test after cleaning.")

    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)

    x_train_scaled = (x_train - mu) / sd
    x_test_scaled = (x_test - mu) / sd

    # Defensive post-scaling cleanup. Near-constant features can create unstable
    # standardized values even when raw model inputs are finite.
    x_train_scaled = np.nan_to_num(x_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_scaled = np.nan_to_num(x_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x_train_scaled = np.clip(x_train_scaled, -10.0, 10.0)
    x_test_scaled = np.clip(x_test_scaled, -10.0, 10.0)

    if not np.isfinite(x_train_scaled).all():
        raise ValueError("Non-finite values found in ridge x_train_scaled.")
    if not np.isfinite(x_test_scaled).all():
        raise ValueError("Non-finite values found in ridge x_test_scaled.")

    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean

    alpha = 10.0
    xtx = np.empty((len(features), len(features)), dtype=float)
    for i in range(len(features)):
        for j in range(len(features)):
            xtx[i, j] = float(np.sum(x_train_scaled[:, i] * x_train_scaled[:, j]))

    xty = np.empty(len(features), dtype=float)
    for i in range(len(features)):
        xty[i] = float(np.sum(x_train_scaled[:, i] * y_centered))

    reg = alpha * np.eye(len(features))

    try:
        coefs = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coefs = np.linalg.pinv(xtx + reg).dot(xty)

    pred_values = np.sum(x_test_scaled * coefs.reshape(1, -1), axis=1) + y_mean

    if not np.isfinite(pred_values).all():
        raise ValueError("Non-finite ridge predictions produced after fitting.")

    pred = pd.Series(pred_values, index=test.index, name="ridge_score")
    importance = pd.Series(coefs, index=features).sort_values(key=lambda x: np.abs(x), ascending=False)
    return ModelResult("ridge", pred, importance)


def fit_random_forest(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, seed: int = 42, n_estimators: int = 40, max_depth: int = 5) -> ModelResult:
    train, test = _clean_model_frame(train, test, features, target)

    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=30, max_depth=max_depth, random_state=seed, n_jobs=-1)
    model.fit(train[features].values, train[target].values)
    pred = pd.Series(model.predict(test[features].values), index=test.index, name="rf_score")
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return ModelResult("random_forest", pred, importance)


def fit_pytorch_mlp(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, epochs: int = 80, hidden_dim: int = 32, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 42, max_train_rows: int = 5000) -> ModelResult:
    """Fit a small PyTorch MLP to predict cross-sectional forward relative returns.

    The default max_train_rows keeps the public smoke test fast and reduces the risk
    of presenting a large neural model as the main research contribution.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)

    if len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=seed).sort_index()

    x_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype("float32")
    y_train = train[target].fillna(0).values.astype("float32").reshape(-1, 1)
    x_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype("float32")

    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    model = nn.Sequential(
        nn.Linear(len(features), hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test)).numpy().reshape(-1)
    return ModelResult("pytorch_mlp", pd.Series(pred, index=test.index, name="mlp_score"), None)
