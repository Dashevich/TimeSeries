from TimeSeriesAnalyser import TimeSeriesAnalysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import statsmodels.api as sm
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional
import logging
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.fbprophet import Prophet as prophet_sktime
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastMSTL
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.performance_metrics.forecasting import MeanAbsoluteError,MeanAbsolutePercentageError
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sklearn.metrics import r2_score
from scipy.stats import ttest_1samp, shapiro, linregress
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from pmdarima import auto_arima
from scipy.fft import fft, fftfreq
from pmdarima.arima import ndiffs


class TimeSeriesClass:
    def __init__(self, store, dates, prices):
        self.store = store
        self.dates = dates
        self.prices = prices
        self.product_ids = self.store.item_id.unique()
        self.forecast_windows = [7, 30, 90]
        warnings.filterwarnings("ignore", category=UserWarning)

    def _get_item_series(self, item_id: str) -> pd.Series:
        """Prepare time series for specific item"""
        item_data = self.store[self.store['item_id'] == item_id].copy()
        item_data['date_id'] = item_data['date_id'].astype(int)

        merged = item_data.merge(
            self.dates[['date_id', 'date']],
            on='date_id',
            how='inner'
        )
        merged['date'] = pd.to_datetime(merged['date'].astype(str))
        merged = merged.set_index('date')['cnt']
        merged.columns = ['target']
        return merged

    def _preprocess_series(self, series: pd.Series) -> pd.Series:
        """Clean and preprocess time series"""
        series = self._remove_leading_zeros(series)
        series = series + 1
        series = self._handle_outliers(series)
        return series

    def _remove_leading_zeros(self, series: pd.Series) -> pd.Series:
        """Remove leading zero values"""
        first_non_zero = series.ne(0).idxmax()
        cnt = int((first_non_zero - series.index[0]).days)
        if cnt >= 100:
          series = series.loc[first_non_zero:]
        return series

    def _handle_outliers(self, series: pd.Series, window: int = 3) -> pd.Series:
        """Detect and replace outliers using rolling median"""
        model = IsolationForest(contamination=0.05)
        outliers = model.fit_predict(series.values.reshape(-1, 1)) == -1

        rolling_median = series.rolling(window, min_periods=1).median()
        series_clean = series.copy()
        series_clean[outliers] = rolling_median[outliers]
        return series_clean

    def hybrid_stl_arima(self, ts, horizon, report):
        ts_series = ts['target'] if isinstance(ts, pd.DataFrame) else ts

        stl = STL(ts_series, period=7).fit()

        trend_model = auto_arima(stl.trend, seasonal=False, suppress_warnings=True)
        trend_fc = trend_model.predict(n_periods=horizon)

        resid_model = auto_arima(stl.resid, seasonal=False, suppress_warnings=True)
        resid_fc = resid_model.predict(n_periods=horizon)

        seasonal = stl.seasonal[-7:].values
        seasonal_fc = np.tile(seasonal, int(np.ceil(horizon/7)))[:horizon]

        return pd.Series(
            np.asarray(trend_fc + resid_fc + seasonal_fc),
            name='forecast'
        )

    def train_sarimax(self, series: pd.Series, horizon: int, report: dict, exog=None, verbose: bool = True):
        """
        Обучает модель SARIMAX на основе параметров из отчета анализа временного ряда.
        """

        d = ndiffs(series, test='adf') if not report['stationarity']['ADF']['stationary'] else 0

        p = report['acf_pacf'].get('suggested_ar_order', 0)
        q = report['acf_pacf'].get('suggested_ma_order', 0)

        seasonal_order = (0, 0, 0, 0)
        if report['seasonality']['exists']:
            s = report['seasonality'].get('period', 12)
            D = ndiffs(series, test='adf', max_d=s) if not report['stationarity']['ADF']['stationary'] else 0
            seasonal_order = (1, D, 1, s)

        max_order = 3
        p = min(p, max_order)
        q = min(q, max_order)
        order = (p, d, q)

        try:
            model = SARIMAX(
                endog=series,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            model_fit = model.fit(disp=0)

            return model_fit.predict(n_periods=horizon)[:horizon]

        except Exception as e:
            print(f"Ошибка при обучении модели: {str(e)}")
            return train_sarimax(series,
                              report={'acf_pacf': {'suggested_ar_order': 1,
                                                  'suggested_ma_order': 1},
                                      'seasonality': {'exists': False},
                                      'stationarity': {'ADF': {'stationary': False}}},
                              horizon=horizon,
                              exog=exog,
                              verbose=verbose)

    def auto_arima_forecast(self, ts, horizon, report=False):
        min_samples = 2 * 7
        seasonal = len(ts) >= min_samples

        model = auto_arima(
            ts,
            seasonal=seasonal,
            m=7 if seasonal else 0,
            stepwise=False,
            trace=report,
            error_action='ignore',
            suppress_warnings=True,
            D=1,
            test='adf',
            seasonal_test='ch',
            scoring='mse',
            n_jobs=-1,
            maxiter=100,
            information_criterion='aicc',
            approximation=False,
            with_intercept=True
        )

        forecast, conf_int = model.predict(
            n_periods=horizon,
            return_conf_int=True
        )

        return forecast

    def calculate_metrics(self, actual, predicted):
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()

        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]

        if len(actual) == 0 or len(predicted) == 0:
            return pd.DataFrame()

        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted)/actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted)**2))

        metrics = {
            'MAE': mae,
            'MAPE (%)': mape,
            'RMSE': rmse,
        }

        return metrics

    def evaluate_models(self, ts, report):
        models = {
            'SARIMAX': self.train_sarimax,
            'AutoARIMA': self.auto_arima_forecast,
            'STL+ARIMA': self.hybrid_stl_arima
        }

        results = {}

        for model_name, model in models.items():
            fig, axes = plt.subplots(3, 1, figsize=(25, 25))
            fig.suptitle(f'Прогнозы модели {model_name}', fontsize=16)

            for idx, horizon in enumerate(self.forecast_windows):
                print(f"Model {model_name} for horizon {horizon}")
                ax = axes[idx]

                forecast = model(ts, horizon, report)
                history_length = min(2*horizon, len(ts))
                history = ts.iloc[-history_length:]

                last_date = history.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date - pd.Timedelta(days=horizon - 1),
                    periods=horizon,
                    freq='D'
                )
                forecast_series = pd.Series(
                    np.asarray(forecast).flatten(),
                    index=forecast_dates,
                    name='forecast'
                )

                ax.plot(history.index, history['target'],
                        label='История')
                ax.plot(forecast_series.index, forecast_series,
                        label='Прогноз', color='red', linewidth=2)

                if forecast_dates[-1] <= ts.index[-1]:
                    actual = ts.loc[forecast_dates[0]:forecast_dates[-1]]
                    ax.plot(actual.index, actual['target'], color = "#1f77b4")

                    metrics = self.calculate_metrics(actual['target'], forecast_series)
                    results[(model_name, horizon)] = metrics

                ax.set_title(f'Горизонт: {horizon} дней')
                ax.legend()
                ax.grid(True)
                ax.set_xlabel('Дата')
                ax.set_ylabel('Продажи')

            plt.tight_layout()
            plt.show()

        return results

    def full_analysis(self, product_id):
        print("Product ", product_id)
        series = self._get_item_series(product_id)
        analyzer = TimeSeriesAnalysis()
        report = analyzer.run_analysis(series)

        clean_series = self._preprocess_series(series)
        print(report)

        if isinstance(clean_series, pd.Series):
          clean_series = clean_series.to_frame(name='target')

        print(self.evaluate_models(clean_series, report))
        return