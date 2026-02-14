"""
ML Models for Predictive Analytics
Includes traffic, pollution, water, and market forecasting
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from typing import Tuple, List
import pandas as pd


class TrafficLSTM(nn.Module):
    """LSTM model for traffic flow prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class PollutionPredictor:
    """XGBoost-based pollution forecasting"""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def feature_importance(self) -> dict:
        return dict(zip(
            [f"feature_{i}" for i in range(len(self.model.feature_importances_))],
            self.model.feature_importances_
        ))


class WaterDemandForecaster:
    """Prophet-based water demand forecasting"""
    
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
    
    def train(self, df: pd.DataFrame):
        # df must have 'ds' (date) and 'y' (value) columns
        self.model.fit(df)
    
    def predict(self, periods: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


class CropYieldPredictor:
    """Random Forest for crop yield prediction"""
    
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ModelEnsemble:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self):
        self.models = {}
    
    def add_model(self, name: str, model):
        self.models[name] = model
    
    def predict(self, X: np.ndarray, weights: List[float] = None) -> np.ndarray:
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred


def create_traffic_model(sequence_length: int = 24) -> TrafficLSTM:
    """Factory function to create traffic prediction model"""
    return TrafficLSTM(
        input_size=10,  # Features: speed, volume, occupancy, weather, etc.
        hidden_size=64,
        num_layers=2,
        output_size=1  # Predicted traffic flow
    )


def create_pollution_model() -> PollutionPredictor:
    """Factory function to create pollution prediction model"""
    return PollutionPredictor()


def create_water_model() -> WaterDemandForecaster:
    """Factory function to create water demand model"""
    return WaterDemandForecaster()
