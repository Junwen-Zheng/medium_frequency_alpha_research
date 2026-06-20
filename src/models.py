from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    name: str
    predictions: pd.Series
    feature_importance: pd.Series | None = None


def fit_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> ModelResult:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(train[features].values, train[target].values)
    pred = pd.Series(model.predict(test[features].values), index=test.index, name="ridge_score")
    coefs = model.named_steps["ridge"].coef_
    importance = pd.Series(coefs, index=features).sort_values(key=lambda x: np.abs(x), ascending=False)
    return ModelResult("ridge", pred, importance)


def fit_random_forest(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, seed: int = 42) -> ModelResult:
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=30, max_depth=5, random_state=seed, n_jobs=-1)
    model.fit(train[features].values, train[target].values)
    pred = pd.Series(model.predict(test[features].values), index=test.index, name="rf_score")
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return ModelResult("random_forest", pred, importance)


def fit_pytorch_mlp(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, epochs: int = 80, hidden_dim: int = 32, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 42) -> ModelResult:
    """Fit a small PyTorch MLP to predict cross-sectional forward relative returns."""
    import torch
    from torch import nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

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
