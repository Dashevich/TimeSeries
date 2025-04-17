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


class TimeSeriesAnalysis:
    def __init__(self, figsize: tuple = (20, 15),
                 acf_lags: int = 40,
                 fourier_freqs: tuple = (1/7, 1/365),
                 alpha: float = 0.05):
        self.figsize = figsize
        self.acf_lags = acf_lags
        self.fourier_freqs = fourier_freqs
        self.alpha = alpha
        self.report = {}

    def run_analysis(self, series: pd.Series) -> Dict[str, Any]:
        """Запускает полный анализ и возвращает отчет"""
        self._validate_input(series)
        self._reset_report()

        self._analyze_trend(series)
        self._analyze_seasonality(series)
        self._analyze_stationarity(series)
        self._analyze_outliers(series)

        cleaned_series = self._preprocess(series)
        self._analyze_acf_pacf(cleaned_series)
        self._analyze_residuals(cleaned_series)
        self._analyze_rolling_stats(cleaned_series)

        self._create_plots(series)
        return self.report

    def _validate_input(self, series):
        """Проверка типа данных"""
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series")

    def _reset_report(self):
        """Сброс отчета"""
        self.report = {
            'trend': None,
            'seasonality': None,
            'stationarity': {},
            'outliers': {},
            'acf_pacf': {},
            'residuals': {},
            'rolling_stats': {}
        }

    def _analyze_trend(self, series):
        """Анализ тренда"""
        time = np.arange(len(series))
        slope, _, _, p_value, _ = linregress(time, series.values)
        self.report['trend'] = {
            'exists': p_value < self.alpha,
            'p_value': p_value,
            'slope': slope
        }

    def _analyze_seasonality(self, series, period=30):
        """Анализ сезонности"""
        try:
            decomp = seasonal_decompose(series.dropna(), period=period)
            seasonal_amplitude = decomp.seasonal.max() - decomp.seasonal.min()
            self.report['seasonality'] = {
                'exists': seasonal_amplitude > 0.1 * series.abs().max(),
                'period': period,
                'amplitude': seasonal_amplitude
            }
        except ValueError:
            self.report['seasonality'] = {
                'exists': False,
                'error': 'Not enough data for decomposition'
            }

    def _analyze_stationarity(self, series):
        """Анализ стационарности"""
        adf_result = adfuller(series.dropna())
        kpss_result = kpss(series.dropna())

        self.report['stationarity'] = {
            'ADF': {
                'p_value': adf_result[1],
                'stationary': adf_result[1] < self.alpha
            },
            'KPSS': {
                'p_value': kpss_result[1],
                'stationary': kpss_result[1] >= self.alpha
            }
        }

    def _analyze_outliers(self, series):
        """Обнаружение выбросов"""
        model = IsolationForest(contamination=0.05)
        outliers = model.fit_predict(series.values.reshape(-1, 1)) == -1
        self.report['outliers'] = {
            'count': sum(outliers),
            'indices': series[outliers].index.tolist()
        }

    def _analyze_acf_pacf(self, series):
        """Анализ ACF/PACF для определения параметров ARIMA"""
        acf_values = acf(series, nlags=self.acf_lags)
        pacf_values = pacf(series, nlags=self.acf_lags, method='ywm')

        critical_value = 1.96 / np.sqrt(len(series))

        significant_acf = np.where(np.abs(acf_values) > critical_value)[0]
        significant_pacf = np.where(np.abs(pacf_values) > critical_value)[0]

        self.report['acf_pacf'] = {
            'significant_acf_lags': significant_acf.tolist(),
            'significant_pacf_lags': significant_pacf.tolist(),
            'suggested_ar_order': significant_pacf[-1] if len(significant_pacf) > 0 else 0,
            'suggested_ma_order': significant_acf[-1] if len(significant_acf) > 0 else 0
        }

    def _analyze_residuals(self, series):
        """Анализ распределения остатков"""
        residuals = series - series.rolling(7).mean()
        _, p_value = shapiro(residuals.dropna())
        self.report['residuals'] = {
            'normal_distribution': p_value > self.alpha,
            'shapiro_p_value': p_value
        }

    def _analyze_rolling_stats(self, series, window=30):
        """Анализ скользящих статистик"""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()

        self.report['rolling_stats'] = {
            'mean_change': abs(rolling_mean.diff().mean()),
            'std_change': abs(rolling_std.diff().mean()),
            'stationary_mean': rolling_mean.std() < 0.1 * series.std(),
            'stationary_std': rolling_std.std() < 0.1 * series.std()
        }

    def _preprocess(self, series):
        """Предобработка данных (очистка и обработка выбросов)"""
        first_non_zero = series.ne(0).idxmax()
        cleaned = series.loc[first_non_zero:]

        if self.report['outliers']['count'] > 0:
            outliers = cleaned.index.isin(self.report['outliers']['indices'])
            cleaned.loc[outliers] = cleaned.rolling(3, min_periods=1).median().loc[outliers]

        return cleaned

    def _create_plots(self, series):
        """Создание комплексного набора графиков"""
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(4, 4)

        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_series(series, ax1, "Исходный временной ряд")

        cleaned_series = self._preprocess(series)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_series(cleaned_series, ax2, "Обработанный ряд")

        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_fourier_spectrum(series, freq=self.fourier_freqs[0], ax=ax3)

        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_fourier_spectrum(series, freq=self.fourier_freqs[1], ax=ax4)

        ax5 = fig.add_subplot(gs[2, :2])
        plot_acf(cleaned_series, lags=self.acf_lags, ax=ax5)

        ax6 = fig.add_subplot(gs[2, 2:])
        plot_pacf(cleaned_series, lags=self.acf_lags, method='ywm', ax=ax6)

        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_error_distribution(cleaned_series, ax=ax7)

        ax8 = fig.add_subplot(gs[3, 1])
        qqplot(cleaned_series, line='s', ax=ax8)

        ax9 = fig.add_subplot(gs[3, 2:])
        self._plot_rolling_stats(cleaned_series, ax=ax9)

    def _preprocess(self, series):
        """Предобработка данных"""
        first_non_zero = series.ne(0).idxmax()
        cleaned = series.loc[first_non_zero:]

        model = IsolationForest(contamination=0.05)
        outliers = model.fit_predict(cleaned.values.reshape(-1, 1)) == -1
        cleaned.loc[outliers] = cleaned.rolling(3, min_periods=1).median().loc[outliers]

        return cleaned + 1

    def _plot_series(self, series, ax, title):
        """Отрисовка временного ряда"""
        series.plot(ax=ax, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.grid(True)

    def _plot_fourier_spectrum(self, series, freq, ax):
        """Спектральный анализ"""
        fft = np.fft.fft(series.values)
        freqs = np.fft.fftfreq(len(series), freq)

        ax.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2])
        ax.set_title(f"Фурье-спектр (1/{int(1/freq)} дней)", fontsize=9)
        ax.grid(True)

    def _plot_error_distribution(self, series, ax):
        """Распределение ошибок"""
        residuals = series - series.rolling(7).mean()
        residuals.plot.hist(ax=ax, bins=20)
        ax.set_title("Распределение ошибок", fontsize=9)

    def _plot_rolling_stats(self, series, ax):
        """Скользящие статистики"""
        rolling_mean = series.rolling(30).mean()
        rolling_std = series.rolling(30).std()

        ax.plot(rolling_mean, label='Скользящее среднее (30 дней)')
        ax.plot(rolling_std, label='Скользящее СКО (30 дней)')
        ax.set_title("Скользящие статистики", fontsize=9)
        ax.legend()
        ax.grid(True)

    def _print_stationarity_report(self, series):
        """Проверка стационарности"""
        adf = adfuller(series)
        kpss_test = kpss(series)

        print("\n" + "="*50)
        print("Анализ стационарности:")
        print(f"ADF тест: p-value = {adf[1]:.4f}")
        print(f"KPSS тест: p-value = {kpss_test[1]:.4f}")
        print("="*50 + "\n")

