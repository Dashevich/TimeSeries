from Processor import *
from Analyser import *
class TimeSeriesClass(TimeSeriesProcessor):
    def __init__(self, store, dates, prices):
        self.store = store
        self.dates = dates
        self.prices = prices
        self.product_ids = self.store.item_id.unique()
        self.forecast_windows = [7, 30, 90]
        warnings.filterwarnings("ignore", category=UserWarning)

    def get_item_series(self, item_id: str) -> pd.Series:
        """Формирует временной ряд продаж для указанного товара."""
        item_data = self.store[self.store['item_id'] == item_id].copy()
        item_data['date_id'] = item_data['date_id'].astype(int)

        merged = item_data.merge(
            self.dates[['date_id', 'date']],
            on='date_id',
            how='left'
        )
        merged['date'] = pd.to_datetime(merged['date'].astype(str))
        merged = merged.set_index('date')['cnt']
        merged.columns = ['target']
        return merged

    def smape_f(self, y_true, y_pred):
      """Вычисляет симметричную среднюю абсолютную процентную ошибку (SMAPE)."""
      epsilon = 1e-8
      numerator = np.abs(y_pred - y_true)
      denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon)
      return 200 * np.mean(numerator / denominator)

    def calculate_metrics(
        self,
        actual: Union[np.ndarray, pd.Series, list],
        predicted: Union[np.ndarray, pd.Series, list]
        ) -> Union[Dict[str, float], pd.DataFrame]:
        """Рассчитывает метрики качества прогноза: MAE, MAPE, RMSE, SMAPE."""
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()

        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]

        if len(actual) == 0 or len(predicted) == 0:
            return pd.DataFrame()

        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        smape = self.smape_f(actual, predicted)
        metrics = {
            'MAE': mae,
            'MAPE (%)': mape,
            'RMSE': rmse,
            'SMAPE': smape,
        }

        return metrics

    from sklearn.linear_model import LinearRegression

    def remove_trend(self, y, return_model=False):
        """Удаляет линейный тренд из ряда"""
        time = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(time, y)
        trend = model.predict(time)
        y_detrended = y - trend
        return (y_detrended, model) if return_model else y_detrended

    def restore_trend(self, y_detrended, model, start=0):
        """Восстанавливает тренд в прогнозных значениях."""
        time = np.arange(start, start + len(y_detrended)).reshape(-1, 1)
        trend = model.predict(time)
        return y_detrended + trend

    def train_xgboost(self, ts, horizon, report):
        """Строит прогнозную модель XGBoost с учетом сезонности и тренда."""
        y = ts['target']
        X = ts.drop(columns=['target']) if ts.shape[1] > 1 else None

        if report['trend'] == 'exists':
            y, trend_model = self.remove_trend(y, return_model=True)

        window_length = report.get('seasonality', {}).get('period', 14)

        regressor = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        forecaster = make_reduction(
            regressor,
            window_length=window_length,
            strategy="recursive"
        )

        forecaster.fit(y, X=X)
        fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
        forecast = forecaster.predict(fh, X=X)
        if report['trend'] == 'exists':
            forecast = self.restore_trend(forecast, trend_model, start=len(y_train))
        return forecast

    def evaluate_models(
        self,
        ts: pd.DataFrame,
        report: Dict[str, Any]
        ) -> Dict[Tuple[str, int], Dict[str, float]]:
        """Оценивает производительность моделей на разных горизонтах прогнозирования."""
        models = {
            'XGBoost': self.train_xgboost,
        }

        results = {}
        for model_name, model in models.items():
            fig, axes = plt.subplots(len(self.forecast_windows), 1, figsize=(25, 25))
            if len(self.forecast_windows) == 1:
                axes = [axes]
            fig.suptitle(f'Прогнозы модели {model_name}', fontsize=16, y=1.01)

            for idx, horizon in enumerate(self.forecast_windows):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                if len(ts) < horizon:
                    continue

                train = ts.iloc[:-horizon]
                test = ts.iloc[-horizon:]

                forecast = model(train, horizon, report)

                forecast_series = pd.Series(
                    forecast.values,
                    index=test.index[:len(forecast)],
                    name='forecast'
                )
                test_target = self.inverse_transform(test['target'])
                train_target = self.inverse_transform(train['target'][-horizon:])
                forecast_series = self.inverse_transform(forecast_series)

                ax.plot(train.index[-horizon:], train_target, label='Тренировочные данные')
                ax.plot(test.index, test_target, label='Тестовые данные', color='green')
                ax.plot(forecast_series.index, forecast_series, label='Прогноз', color='red', linewidth=2)

                metrics = self.calculate_metrics(test['target'], forecast_series)
                results[(model_name, horizon)] = metrics

                ax.set_title(f'Горизонт: {horizon} дней')
                ax.legend()
                ax.grid(True)
                ax.set_xlabel('Дата')
                ax.set_ylabel('Продажи')

            plt.tight_layout()
            plt.show()

        return results

    def full_analysis(self, product_id: str, show_plots: bool = False) -> None:
        """Выполняет комплексный анализ и прогнозирование для указанного товара."""
        print("Product ", product_id)
        series = self.get_item_series(product_id)
        analyzer = TimeSeriesAnalysis()
        report = analyzer.run_analysis(series, show_plots)

        clean_series = self.preprocess(series)
        print(report)

        if isinstance(clean_series, pd.Series):
            clean_series = clean_series.to_frame(name='target')

        res = (self.evaluate_models(clean_series, report))
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(res)
        return