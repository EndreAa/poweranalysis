""" 
pythonskript for å bygge en hdd modell. 
https://en.wikipedia.org/wiki/Heating_degree_day
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


class HDDModel:
    """
    Heating Degree Day regression model.
    """

    def __init__(
        self,
        base_temp: float | None = None,
        temp_col: str = "air_temperature",
        target_col: str = "Forbruk (kWh)",
    ) -> None:
        
        self.base_temp = base_temp
        self.temp_col = temp_col
        self.target_col = target_col
        self._model: object | None = None

    @staticmethod
    def compute_hdd(temp: pd.Series, base_temp: float) -> pd.Series:
        return np.clip(base_temp - temp, 0, None)

    def find_optimal_base(self, df: pd.DataFrame, candidates=None) -> float:
        if candidates is None:
            candidates = np.arange(10, 21, 0.5)

        best_r2 = -np.inf
        best_T = float(candidates[0])
        y = df[self.target_col]

        for T in candidates:
            HDD = self.compute_hdd(df[self.temp_col], T)
            X = sm.add_constant(HDD)
            m = sm.OLS(y, X, missing="drop").fit()
            if m.rsquared > best_r2:
                best_r2 = m.rsquared
                best_T = float(T)

        self.base_temp = best_T
        return best_T

    def fit(self, df: pd.DataFrame) -> object:
        if self.base_temp is None:
            raise ValueError("base_temp not set. Run find_optimal_base first or set manually.")

        HDD = self.compute_hdd(df[self.temp_col], self.base_temp)
        X = sm.add_constant(HDD)
        y = df[self.target_col]

        self._model = sm.OLS(y, X, missing="drop").fit()
        return self._model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        HDD = self.compute_hdd(df[self.temp_col], self.base_temp)
        X = sm.add_constant(HDD)
        return self._model.predict(X)

    def summary(self):
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.summary()