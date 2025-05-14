import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import statsmodels.api as sm
from scipy import stats
from typing import List, Tuple, Union, Dict, Any, Optional
import logging

from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastMSTL
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.performance_metrics.forecasting import MeanAbsoluteError,MeanAbsolutePercentageError
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.forecasting.compose import make_reduction, TransformedTargetForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.split import temporal_train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest

from scipy.fft import fft, fftfreq
from scipy.stats import ttest_1samp, shapiro, linregress
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima
from pmdarima.arima import ndiffs
from matplotlib.axes import Axes
from xgboost import XGBRegressor
import pprint
class TimeSeriesProcessor:
    @staticmethod
    def remove_leading_zeros(series: pd.Series) -> pd.Series:
        """Удаляет ведущие нулевые значения из временного ряда"""     
        first_non_zero = series.ne(0).idxmax()
        cnt = int((first_non_zero - series.index[0]).days)
        if cnt >= 100:
            return series.loc[first_non_zero:]
        return series

    @staticmethod
    def handle_outliers(series: pd.Series, window: int = 3) -> pd.Series:
        """Обнаруживает и заменяет выбросы"""
        model = IsolationForest(contamination=0.05)
        outliers = model.fit_predict(series.values.reshape(-1, 1)) == -1
        rolling_median = series.rolling(window, min_periods=1).median()
        return series.mask(outliers, rolling_median)

    @staticmethod
    def inverse_transform(forecast: pd.Series) -> pd.Series:
        """Отмена преобразований и обработка отрицательных значений"""
        forecast = forecast - 1
        forecast[forecast < 0] = 0
        return forecast

    def preprocess(self, series: pd.Series) -> pd.Series:
        """Предобработка данных с сохранением параметров трансформации"""
        cleaned = self.remove_leading_zeros(series)
        cleaned = self.handle_outliers(cleaned)
        cleaned[cleaned < 0] = 0
        cleaned = cleaned + 1
        return cleaned