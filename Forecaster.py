
class TimeSeriesClass(TimeSeriesProcessor):
    def __init__(self, store, dates, prices):
        self.store = store
        self.dates = dates
        self.prices = prices
        self.product_ids = self.store.item_id.unique()
        self.forecast_windows = [7, 30, 90]
        self.models_path = ""
        self.forecaster = None
        self.trend_model = None
        self.training_length = None
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

    def fit(self, ts, horizon, report):
        """Обучение модели XGBoost с учетом тренда и сезонности."""
        y = ts['target']
        X = ts.drop(columns=['target']) if ts.shape[1] > 1 else None

        # Удаление тренда при необходимости
        self.trend_model = None
        if report.get('trend') == 'exists':
            y, self.trend_model = self.remove_trend(y, return_model=True)

        # Создание оконных признаков
        window_length = report.get('seasonality', {}).get('period', 14)
        regressor = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.forecaster = make_reduction(
            regressor,
            window_length=window_length,
            strategy="recursive"
        )

        # Обучение модели
        self.forecaster.fit(y, X=X)
        return self.forecaster, self.trend_model

    def save_model(self, item_id, horizon):
        """Сохранение модели и связанных параметров."""
        model_dict = {
            'forecaster': self.forecaster,
            'trend_model': self.trend_model,
        }
        model_path = os.path.join(self.models_path, f"xgboost_{item_name}.pkl")
        joblib.dump(model_dict, model_path)
        return model_path

    def load_model(self, model_path, item_id):
        """Загрузка сохраненной модели."""
        model_path = os.path.join(self.models_path, f"xgboost_{item_id}.pkl")
        model_dict = joblib.load(model_path)
        self.forecaster = model_dict['forecaster']
        self.trend_model = model_dict['trend_model']
        return self.forecaster, self.trend_model

    def predict(self, horizon, X=None):
        """Прогнозирование с использованием обученной модели."""
        if self.forecaster is None:
            raise ValueError("Модель не загружена. Сначала обучите или загрузите модель.")

        fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
        forecast = self.forecaster.predict(fh, X=X)

        if self.trend_model is not None and self.training_length is not None:
            forecast = self.restore_trend(forecast, self.trend_model, start=self.training_length)
        return forecast

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
                hor_max = max(forecast_windows)
                train = ts.iloc[:-hor_max]
                test = ts.iloc[-hor_max:horizon]

                forecast = model(train, horizon, report)

                forecast_series = pd.Series(
                    forecast.values,
                    index=test.index[:len(forecast)],
                    name='forecast'
                )
                test_target = self.inverse_transform(test['target'])
                train_target = self.inverse_transform(train['target'][-horizon:])
                forecast_series = self.inverse_transform(forecast_series)

        def show_info(y_true, y_pred, item_id):
            ax.plot(y_true.index, y_true['target'], label='Тестовые данные', color='green')
            ax.plot(y_pred.index, y_pred, label='Прогноз', color='red', linewidth=2)

            metrics = self.calculate_metrics(y_true['target'], y_pred)
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