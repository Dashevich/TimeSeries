from Processor import *

class TimeSeriesAnalysis(TimeSeriesProcessor):
    def __init__(self, figsize: tuple = (20, 15), acf_lags: int = 40,
                 fourier_freqs: tuple = (1/7, 1/365), alpha: float = 0.05) -> None:
        super().__init__()
        self.figsize = figsize
        self.acf_lags = acf_lags
        self.fourier_freqs = fourier_freqs
        self.alpha = alpha
        self.report: Dict[str, Any] = {}

    def run_analysis(self, series: pd.Series, show_plots: bool = False) -> Dict[str, Any]:
        """Запускает полный анализ и возвращает отчет"""
        self._reset_report()

        self._analyze_trend(series)
        self._analyze_seasonality(series)
        self._analyze_stationarity(series)
        self._analyze_outliers(series)
        cleaned_series = self.preprocess(series)
        if show_plots:
          self._create_plots(series)
        return self.report

    def _reset_report(self) -> None:
        """Сброс отчета"""
        self.report = {
            'trend': None,
            'seasonality': None,
            'stationarity': {},
            'outliers': {},
        }

    def _analyze_trend(self, series: pd.Series) -> None:
        """Анализ тренда"""
        time = np.arange(len(series))
        slope, _, _, p_value, _ = linregress(time, series.values)
        self.report['trend'] = {
            'exists': p_value < self.alpha,
            'p_value': p_value,
            'slope': slope
        }

    def _analyze_seasonality(self, series: pd.Series, period: int = 30) -> None:
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

    def _analyze_stationarity(self, series: pd.Series) -> None:
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

    def _analyze_outliers(self, series: pd.Series) -> None:
        """Обнаружение выбросов"""
        model = IsolationForest(contamination=0.05)
        outliers = model.fit_predict(series.values.reshape(-1, 1)) == -1
        self.report['outliers'] = {
            'count': sum(outliers),
            'indices': series[outliers].index.tolist()
        }

    def _create_plots(self, series: pd.Series) -> None:
        """Создание комплексного набора графиков"""
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 4)

        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_series(series, ax1, "Исходный временной ряд")

        cleaned_series = self.preprocess(series)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_series(cleaned_series, ax2, "Обработанный ряд")

        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_fourier_spectrum(series, freq=self.fourier_freqs[0], ax=ax3)

        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_fourier_spectrum(series, freq=self.fourier_freqs[1], ax=ax4)

        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_error_distribution(cleaned_series, ax=ax7)

        ax8 = fig.add_subplot(gs[2, 1])
        qqplot(cleaned_series, line='s', ax=ax8)

    def _plot_series(self, series: pd.Series, ax: Axes, title: str) -> None:
        """Отрисовка временного ряда"""
        series.plot(ax=ax, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.grid(True)

    def _plot_fourier_spectrum(self, series: pd.Series, freq: float, ax: Axes) -> None:
        """Спектральный анализ"""
        fft = np.fft.fft(series.values)
        freqs = np.fft.fftfreq(len(series), freq)

        ax.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2])
        ax.set_title(f"Фурье-спектр (1/{int(1/freq)} дней)", fontsize=9)
        ax.grid(True)

    def _plot_error_distribution(self, series: pd.Series, ax: Axes) -> None:
        """Распределение ошибок"""
        residuals = series - series.rolling(7).mean()
        residuals.plot.hist(ax=ax, bins=20)
        ax.set_title("Распределение ошибок", fontsize=9)

    def _print_stationarity_report(self, series: pd.Series) -> None:
        """Проверка стационарности"""
        adf = adfuller(series)
        kpss_test = kpss(series)

        print("\n" + "="*50)
        print("Анализ стационарности:")
        print(f"ADF тест: p-value = {adf[1]:.4f}")
        print(f"KPSS тест: p-value = {kpss_test[1]:.4f}")
        print("="*50 + "\n")