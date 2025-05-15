from Processor import *
from Analyser import *

class TimeSeriesClass(TimeSeriesProcessor):
    def __init__(self, store, dates, prices, models_path):
        self.store = store
        self.dates = dates
        self.prices = prices
        self.product_ids = self.store.item_id.unique()
        self.forecast_windows = [7, 30, 90]
        self.report = {}
        self.models_path = models_path
        warnings.filterwarnings("ignore", category=UserWarning)

    def preprocess_item(self, item_id: str) -> pd.Series:
        """Формирует временной ряд продаж для указанного товара."""
        features = self.add_features()
        item_data = features[(features.item_id == item_id)]
        item_data['date_id'] = item_data['date_id'].astype(int)
        merged = item_data.merge(
            self.store[['cnt', 'date_id', 'item_id']], 
            how='left',
            on=['date_id', 'item_id']
        )
        merged.drop(['store_id', 'item_id', 'date_id', 'wm_yr_wk', 'year', 'month', 'index'], axis=1, inplace=True)

        merged['date'] = pd.to_datetime(merged['date'].astype(str))
        merged = merged.set_index('date')
        merged.rename(columns={'cnt': 'target'}, inplace=True)

        analyzer = TimeSeriesAnalysis()
        report = analyzer.run_analysis(merged['target'], show_plots=False)

        merged['target'] = self.preprocess(merged['target'])
        if isinstance(merged, pd.Series):
            merged = merged.to_frame(name='target')
        return merged

    def add_features(self) -> pd.DataFrame:
        """Объединяет данные дат и цен в единый DataFrame."""
        new_merged = self.dates.merge(self.prices, how='left', on="wm_yr_wk")
        return new_merged

    def smape_f(
        self,
        actual: Union[np.ndarray, pd.Series, list],
        predicted: Union[np.ndarray, pd.Series, list]
        ) -> float:
      """Вычисляет симметричную среднюю абсолютную процентную ошибку (SMAPE)."""
      epsilon = 1e-8
      numerator = np.abs(predicted - actual)
      denominator = (np.abs(actual) + np.abs(predicted) + epsilon)
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
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        smape = self.smape_f(actual, predicted)
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'SMAPE': smape,
        }

        return metrics

    def fit(self, ts: pd.DataFrame) -> tuple:
        """Обучение модели CatBoost с учетом тренда и сезонности."""
        y = ts['target']
        X = ts.drop(columns=['target']) if ts.shape[1] > 1 else None

        trend_model = None
        if self.report.get('trend') == 'exists':
            y, trend_model = self.remove_trend(y)

        category_encoder = None
        if X is not None:
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 0:
                category_encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1
                )
                X[cat_cols] = category_encoder.fit_transform(X[cat_cols])

        window_length = self.report.get('seasonality', {}).get('period', 14)
        regressor = CatBoostRegressor(
            iterations=500,      
            depth=5,            
            learning_rate=0.1,
            subsample=0.8,
            random_seed=42,
            verbose=0      
        )
        forecaster = make_reduction(
            regressor,
            window_length=window_length,
            strategy="recursive"
        )

        forecaster.fit(y, X=X)
        return (forecaster, category_encoder), trend_model

    def save_model(self, item_id: str, forecaster: Any, trend_model: Any) -> None:
        """Сохранение модели и связанных параметров."""
        os.makedirs(self.models_path, exist_ok=True)
        model_dict = {
            'forecaster': forecaster,
            'trend_model': trend_model,
        }
        model_path = os.path.join(self.models_path, f"xgboost_{item_id}.pkl")
        joblib.dump(model_dict, model_path)

    def load_model(self, item_id: str) -> tuple:
        """Загрузка сохраненной модели."""
        model_path = os.path.join(self.models_path, f"xgboost_{item_id}.pkl")
        model_dict = joblib.load(model_path)
        forecaster = model_dict['forecaster']
        trend_model = model_dict['trend_model']
        return forecaster, trend_model

    def predict(self, X: pd.DataFrame, forecaster: tuple = None, trend_model: Any = None) -> pd.Series:
        """Прогнозирование с использованием обученной модели и обработкой категориальных признаков."""
        forecaster, category_encoder = forecaster
        if forecaster is None:
            raise ValueError("Модель не загружена. Сначала обучите или загрузите модель.")
        if X is not None and category_encoder is not None:
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(cat_cols) > 0:
                X = X.copy()
                X[cat_cols] = category_encoder.transform(X[cat_cols])

        horizon = len(X)
        fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
        
        forecast = forecaster.predict(fh, X=X)
        if trend_model is not None:
            forecast = self.restore_trend(forecast, trend_model)
        forecast_series = pd.Series(
            forecast.values,
            index=X.index[:len(forecast)],  
            name='forecast'
        )
        
        forecast_series = self.inverse_transform(forecast_series)
        
        return forecast_series

    def remove_trend(self, y: pd.Series) -> tuple:
        """Удаляет линейный тренд из ряда"""
        time = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(time, y)
        trend = model.predict(time)
        y_detrended = y - trend
        return (y_detrended, model)

    def restore_trend(self, y_detrended: pd.Series, model: Any, start: int = 0) -> pd.Series:
        """Восстанавливает тренд в прогнозных значениях."""
        time = np.arange(start, start + len(y_detrended)).reshape(-1, 1)
        trend = model.predict(time)
        return y_detrended + trend

    def train_test_split(self, ts: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Разделение данных на тестовые и тренировочные"""
        hor_max = max(self.forecast_windows)
        train = ts.iloc[:-hor_max]
        test = ts.iloc[-hor_max:]
        test = test.iloc[:horizon]
        return train, test

    def show_info(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """Визуализация данных и сбор метрик"""
        y_true = self.inverse_transform(y_true)
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.plot(y_true.index, y_true, label='Тестовые данные', color='green')
        ax.plot(y_pred.index, y_pred, label='Прогноз', color='red', linewidth=2)
        results = {}
        metrics = self.calculate_metrics(y_true, y_pred)

        ax.set_title(f'Горизонт: {len(y_true)} дней')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Дата')
        ax.set_ylabel('Продажи')

        plt.tight_layout()
        plt.show()
        return metrics