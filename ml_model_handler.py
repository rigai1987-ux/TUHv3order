import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
import optuna, streamlit as st, torch
from trading_simulator import find_bracket_entry, find_first_exit
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- НОВЫЙ БЛОК: Определение модели нейронной сети ---
class SimpleMLP(nn.Module):
    """Простой многослойный перцептрон (MLP) для бинарной классификации."""
    def __init__(self, input_size, hidden_size=64, num_hidden_layers=2, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        layers = []
        # Входной слой
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        # Скрытые слои
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        # Выходной слой
        layers.append(nn.Linear(hidden_size, 1)) # Один выход для бинарной классификации
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_trade_outcome(
    signal_idx: int,
    params: dict,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    hldir_values: np.ndarray
) -> int:
    """
    Определяет исход одного сигнала (TP, SL или нет входа) для разметки.
    Эта функция полностью имитирует логику симулятора для одного сигнала.

    Args:
        signal_idx: Индекс сигнальной свечи.
        params: Словарь с параметрами.
        ... массивы цен ...

    Returns:
        1: если сделка закрылась по Take Profit.
        0: если сделка закрылась по Stop Loss или по таймауту разметки.
        -1: если входа в сделку не произошло.
    """
    # --- 1. Получаем параметры, идентичные симулятору ---
    bracket_offset_pct = params.get("bracket_offset_pct", 0.5)
    bracket_timeout_candles = params.get("bracket_timeout_candles", 5)
    stop_loss_pct = params.get('stop_loss_pct', 2.0)
    take_profit_pct = params.get('take_profit_pct', 4.0)
    labeling_timeout = params.get('ml_labeling_timeout_candles', 20)
    use_hldir_on_conflict = params.get("use_hldir_on_conflict", True)
    simulate_slippage = params.get("simulate_slippage", True)

    # --- 2. Ищем вход в сделку (логика идентична trading_simulator.py) ---
    base_price = open_prices[signal_idx + 1]
    long_level = base_price * (1 + bracket_offset_pct / 100)
    short_level = base_price * (1 - bracket_offset_pct / 100)

    entry_idx, entry_price, direction_str = find_bracket_entry(
        start_idx=signal_idx + 1,
        timeout=bracket_timeout_candles,
        long_level=long_level,
        short_level=short_level,
        use_hldir_on_conflict=use_hldir_on_conflict,
        simulate_slippage=simulate_slippage,
        hldir_values=hldir_values,
        high_prices=high_prices,
        low_prices=low_prices,
        open_prices=open_prices
    )

    if direction_str == "none":
        return -1 # Входа не было

    # --- 3. Ищем выход из сделки (логика идентична trading_simulator.py) ---
    direction_is_long = (direction_str == 'long')
    stop_loss_price = entry_price * (1 - stop_loss_pct / 100) if direction_is_long else entry_price * (1 + stop_loss_pct / 100)
    take_profit_price = entry_price * (1 + take_profit_pct / 100) if direction_is_long else entry_price * (1 - take_profit_pct / 100)

    exit_idx, exit_reason, _ = find_first_exit(
        entry_idx=entry_idx,
        direction_is_long=direction_is_long,
        stop_loss=stop_loss_price,
        take_profit=take_profit_price,
        high_prices=high_prices, low_prices=low_prices, close_prices=close_prices
    )

    # --- 4. Определяем метку на основе исхода ---
    if exit_idx > entry_idx and exit_idx < entry_idx + labeling_timeout:
        return 1 if exit_reason == 'take_profit' else 0
    else:
        # Если выход произошел за пределами окна разметки или в конце данных, считаем это неудачей
        return 0

def _label_single_signal_wrapper(signal_idx, params, open_prices, high_prices, low_prices, close_prices, hldir_values, df_len):
    """
    Обертка для параллельной обработки одного сигнала.
    Возвращает (индекс, исход) или None.
    """    
    # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Пропускаем сигналы слишком близко к концу данных ---
    # Это гарантирует, что окно разметки (`ml_labeling_timeout_candles`)
    # не выйдет за пределы предоставленного DataFrame (т.е. за пределы train_data в WFO).
    required_future_bars = params.get("bracket_timeout_candles", 5) + params.get('ml_labeling_timeout_candles', 20)
    if signal_idx + required_future_bars >= df_len:
        return None

    outcome = get_trade_outcome(signal_idx, params, open_prices, high_prices, low_prices, close_prices, hldir_values)
    if outcome != -1: # -1 означает, что входа не было
        return (signal_idx, outcome)
    return None

def label_all_signals(df_with_features: pd.DataFrame, signal_indices: list, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Размечает ВСЕ СИГНАЛЫ на "успешные" (1) и "провальные" (0) для обучения ML-модели.
    Для каждого сигнала выполняется микро-симуляция, чтобы определить его исход.

    Args:
        df_with_features (pd.DataFrame): DataFrame с рассчитанными признаками.
        signal_indices (list): Список индексов свечей, на которых сгенерированы сигналы.
        params (dict): Словарь с параметрами стратегии и разметки.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Кортеж, содержащий:
            - X (pd.DataFrame): DataFrame с признаками для обучения.
            - y (pd.Series): Series с целевыми метками (0 или 1).
    """
    # Извлекаем массивы numpy для быстрой передачи в параллельные задачи
    open_prices = df_with_features['open'].values
    high_prices = df_with_features['high'].values
    low_prices = df_with_features['low'].values
    close_prices = df_with_features['close'].values
    hldir_values = df_with_features['HLdir'].values if 'HLdir' in df_with_features.columns else np.zeros_like(open_prices)
    df_len = len(df_with_features)

    # --- УЛУЧШЕНИЕ: Параллельная разметка сигналов ---
    # Используем все доступные ядра CPU. backend='threading' хорошо подходит для I/O-bound или
    # смешанных задач, где Numba/Numpy могут освобождать GIL.
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(_label_single_signal_wrapper)(idx, params, open_prices, high_prices, low_prices, close_prices, hldir_values, df_len)
        for idx in signal_indices
    )

    # Фильтруем None и разделяем результаты
    labeled_signals = [res for res in results if res is not None]
    if not labeled_signals:
        return pd.DataFrame(), pd.Series(dtype='int')
    valid_signal_indices, labels = zip(*labeled_signals)

    if not valid_signal_indices:
        return pd.DataFrame(), pd.Series(dtype='int')

    # --- ИСПРАВЛЕНИЕ: Преобразуем кортеж valid_signal_indices в список для iloc ---
    # iloc ожидает список для индексации строк, а zip возвращает кортеж.
    # Это исправляет ошибку "Too many indexers".
    valid_indices_list = list(valid_signal_indices)
    X = df_with_features.iloc[valid_indices_list].copy()
    # Создаем y с тем же индексом, что и у X, для корректного выравнивания.
    y = pd.Series(labels, index=X.index)

    return X, y

def generate_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Обогащает DataFrame дополнительными признаками для ML-модели.

    Args:
        df (pd.DataFrame): Исходный DataFrame с индикаторами.
        params (dict): Словарь с параметрами.

    Returns:
        pd.DataFrame: DataFrame с добавленными ML-признаками.
    """
    df_copy = df.copy()

    # 1. Признаки на основе принтов и рыночной силы (пример, требует адаптации)
    # Эти расчеты предполагают, что у вас есть столбцы 'long_prints', 'short_prints', 'LongM', 'ShortM'
    # Если их нет, эти строки нужно закомментировать или адаптировать.
    prints_window = params.get('ml_prints_window', 10)
    df_copy['prints_strength'] = (df_copy['long_prints'].rolling(window=prints_window).sum() - 
                                  df_copy['short_prints'].rolling(window=prints_window).sum()).astype(float)
    df_copy['market_strength'] = (df_copy['LongM'].rolling(window=prints_window).sum() - 
                                  df_copy['ShortM'].rolling(window=prints_window).sum()).astype(float)

    # 2. Признаки на основе свечей
    df_copy['hldir_strength'] = df_copy['HLdir'].rolling(window=prints_window).mean().astype(float)
    body_size = abs(df_copy['close'] - df_copy['open'])
    candle_range = (df_copy['high'] - df_copy['low']).replace(0, np.nan)
    df_copy['relative_body_size'] = (body_size / candle_range).astype(float)
    df_copy['upper_wick_ratio'] = ((df_copy['high'] - np.maximum(df_copy['open'], df_copy['close'])) / candle_range).astype(float)
    # --- ИСПРАВЛЕНИЕ: Явно приводим growth_pct к типу float ---
    # Это гарантирует, что столбец будет числовым, даже если расчеты приведут к NaN или inf.
    df_copy['growth_pct'] = (df_copy['close'].pct_change(periods=params.get('lookback_period', 20)) * 100).astype(float)

    # 3. Временные признаки
    if 'datetime' in df_copy.columns:
        # --- ИСПРАВЛЕНИЕ: Явно приводим тип к int ---
        # Это предотвращает создание столбца с dtype='object', если df_copy['datetime'] содержит пропуски (NaT),
        # и решает проблему TypeError при конвертации в тензор PyTorch.
        # Сначала принудительно преобразуем в datetime, обрабатывая ошибки (errors='coerce' заменит неверные форматы на NaT).
        # Затем извлекаем час и безопасно приводим к int, заполняя пропуски.
        datetimes = pd.to_datetime(df_copy['datetime'], errors='coerce')
        df_copy['hour_of_day'] = datetimes.dt.hour.astype('Int64').fillna(0).astype(int)

    # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: Гарантируем, что ВСЕ столбцы являются числовыми ---
    # Принудительно преобразуем все столбцы в числовой формат. errors='coerce' заменит любые
    # нечисловые значения (например, оставшиеся строки) на NaN.
    # Это решает проблему, если исходный df из generate_signals уже содержал столбцы типа 'object'.
    df_numeric = df_copy.apply(pd.to_numeric, errors='coerce')
    # Заполняем все NaN и inf, которые могли возникнуть на любом из этапов.
    return df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

def prepare_data_for_nn(X: pd.DataFrame, model_bundle: dict, is_training: bool = True) -> pd.DataFrame:
    """
    Централизованная функция для подготовки данных для нейронной сети.
    Выполняет масштабирование, one-hot encoding и приведение типов.

    Args:
        X (pd.DataFrame): Входной DataFrame с признаками.
        model_bundle (dict): Словарь, содержащий 'scaler', 'numerical_features' и т.д.
        is_training (bool): Флаг, указывающий, находится ли функция в режиме обучения.

    Returns:
        pd.DataFrame: DataFrame, готовый для преобразования в тензор.
    """
    scaler = model_bundle['scaler']
    numerical_features = model_bundle['numerical_features']
    categorical_features = model_bundle.get('categorical_features', [])

    X_scaled = X.copy()
    if scaler and numerical_features:
        num_features_in_df = [f for f in numerical_features if f in X_scaled.columns]
        if num_features_in_df:
            transform_method = scaler.fit_transform if is_training else scaler.transform
            X_scaled[num_features_in_df] = pd.DataFrame(transform_method(X_scaled[num_features_in_df]), index=X_scaled.index, columns=num_features_in_df)

    X_nn = pd.get_dummies(X_scaled, columns=categorical_features, drop_first=True, dtype=float)

    if not is_training and 'nn_columns' in model_bundle:
        X_nn = X_nn.reindex(columns=model_bundle['nn_columns'], fill_value=0)

    # Финальная очистка, чтобы гарантировать 100% числовой формат
    X_nn_numeric = X_nn.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X_nn_numeric

def train_ml_model(X: pd.DataFrame, y: pd.Series, ml_params: dict, model_type: str = 'CatBoost'):
    """
    Обучает ML-модель (CatBoost или Нейросеть) на предоставленных данных.
    
    Args:
        X (pd.DataFrame): DataFrame с признаками.
        y (pd.Series): Series с метками (0 или 1).
        ml_params (dict): Словарь с гиперпараметрами для CatBoost.
        model_type (str): Тип модели для обучения ('CatBoost' или 'NeuralNetwork').

    Returns:
        dict: Словарь, содержащий обученную модель, скейлер, имена признаков и их важность.
    """
    # Определяем список всех признаков, которые будут использоваться
    feature_names = [
        'prints_strength', 'market_strength', 'hldir_strength',
        'relative_body_size', 'upper_wick_ratio',
        'hour_of_day',
        'natr', 'growth_pct' # Добавляем базовые индикаторы как признаки
    ]
    
    # Отбираем только те признаки, которые есть в DataFrame X
    available_features = [f for f in feature_names if f in X.columns]
    X_train = X[available_features]
    # Определяем категориальные и числовые признаки
    categorical_features = [col for col in ['hour_of_day'] if col in X_train.columns]
    numerical_features = [col for col in available_features if col not in categorical_features]
    
    # Масштабирование числовых признаков
    scaler = StandardScaler()
    model = None
    feature_importances = pd.DataFrame()

    if model_type == 'CatBoost':
        # CatBoost ожидает, что категориальные признаки будут в исходном виде (немасштабированные)
        # Поэтому мы передаем DataFrame, где только числовые признаки были масштабированы.
        # Инициализация и обучение модели
        model = CatBoostClassifier(
            iterations=ml_params.get('ml_iterations', 300),
            depth=ml_params.get('ml_depth', 4),
            learning_rate=ml_params.get('ml_learning_rate', 0.1),
            verbose=0,  # Отключаем логирование обучения в консоль
            random_seed=42,
            cat_features=categorical_features
        )
        # Для CatBoost масштабируем только числовые признаки
        X_train_catboost = X_train.copy()
        if numerical_features:
            X_train_catboost[numerical_features] = scaler.fit_transform(X_train_catboost[numerical_features])

        model.fit(X_train_catboost, y)
        
        # Получаем важность признаков для CatBoost
        feature_importances = pd.DataFrame({
            'feature': available_features,
            'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)

    elif model_type == 'NeuralNetwork':
        # --- ИСПОЛЬЗУЕМ НОВУЮ ЦЕНТРАЛИЗОВАННУЮ ФУНКЦИЮ ---
        # Создаем временный бандл для передачи в функцию подготовки
        temp_bundle = {
            "scaler": scaler,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features
        }
        X_nn = prepare_data_for_nn(X_train, temp_bundle, is_training=True)
        available_features = X_nn.columns.tolist()

        # Преобразуем данные в тензоры PyTorch
        X_tensor = torch.tensor(X_nn.values.astype(np.float32), dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        # Создаем DataLoader для пакетного обучения
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=ml_params.get('ml_batch_size', 64), shuffle=True)

        # Инициализируем модель, функцию потерь и оптимизатор
        input_size = X_nn.shape[1]
        model = SimpleMLP(
            input_size=input_size,
            hidden_size=ml_params.get('ml_hidden_size', 64),
            num_hidden_layers=ml_params.get('ml_num_hidden_layers', 2),
            dropout_rate=ml_params.get('ml_dropout_rate', 0.5)
        )
        criterion = nn.BCEWithLogitsLoss() # Функция потерь для бинарной классификации
        optimizer = optim.Adam(model.parameters(), lr=ml_params.get('ml_learning_rate', 0.001))

        # Цикл обучения
        epochs = ml_params.get('ml_epochs', 20)
        model.train() # Переводим модель в режим обучения
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Для нейросетей нет простого способа получить 'feature_importance' как в CatBoost.
        # Оставим эту часть пустой или можно реализовать более сложные методы (Permutation Importance).
        feature_importances = pd.DataFrame(columns=['feature', 'importance', 'description'])

    # --- Общий код для обеих моделей ---
    if not feature_importances.empty:
        # --- НОВЫЙ БЛОК: Создание словаря с описаниями признаков ---
        prints_window = ml_params.get('ml_prints_window', 10)
        feature_descriptions = {
            # --- ИСПРАВЛЕНИЕ: Используем f-string для динамического отображения периода ---
            'prints_strength': f"**Сила принтов (окно {prints_window})**: Суммарная разница между 'длинными' и 'короткими' принтами.",
            'market_strength': f"**Сила рынка (окно {prints_window})**: Суммарная разница между рыночными покупками и продажами.",
            'hldir_strength': f"**Сила направления свечи (окно {prints_window})**: Среднее значение `HLdir` (положение закрытия относительно high/low).",
            'relative_body_size': "**Относительный размер тела**: Отношение размера тела свечи к её общему диапазону. Формула: `abs(close - open) / (high - low)`",
            'upper_wick_ratio': "**Отношение верхней тени**: Отношение верхней тени к общему диапазону свечи. Формула: `(high - max(open, close)) / (high - low)`",
            'hour_of_day': "**Час дня**: Час, в который сформировался сигнал (0-23). Категориальный признак.",
            'natr': "**Нормализованный ATR**: Индикатор волатильности NATR. Показывает средний диапазон свечи в процентах от цены.",
            'growth_pct': "**Процент роста**: Рост цены за `lookback_period` свечей. Показывает недавний импульс."
        }
        # --- Конец нового блока ---
        feature_importances['description'] = feature_importances['feature'].map(feature_descriptions)

    # Возвращаем "бандл" с моделью и всем необходимым для предсказаний
    model_bundle = {
        "model_type": model_type, # Сохраняем тип модели
        "model": model,
        "scaler": scaler,
        "feature_names": available_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features, # Добавляем информацию о категориальных признаках
        "feature_importances": feature_importances,
        # Для нейросети сохраняем one-hot encoded колонки, чтобы использовать их при предсказании
        "nn_columns": X_nn.columns.tolist() if model_type == 'NeuralNetwork' else None
    }
    
    return model_bundle

def run_ml_params_optimization(bundle: dict):
    """
    Запускает Optuna для поиска лучших гиперпараметров ML-модели и признаков.

    Args:
        bundle (dict): Словарь, содержащий все необходимое для оптимизации:
            - df_with_indicators: DataFrame с базовыми индикаторами.
            - signal_indices: Исходные индексы сигналов.
            - base_params: Базовые параметры стратегии.
            - param_ranges: Диапазоны для оптимизируемых ML-параметров.
            - n_trials: Количество проб Optuna.
            - target_metric: Метрика для оптимизации ('F1-Score', 'Precision', etc.).
    """
    df_with_indicators = bundle['df_with_indicators']
    signal_indices = bundle['signal_indices']
    base_params = bundle['base_params']
    param_ranges = bundle['param_ranges']
    n_trials = bundle['n_trials']
    target_metric = bundle['target_metric']

    # Определяем метрику
    metric_func = {
        "F1-Score": f1_score,
        "Precision": precision_score,
        "Accuracy": accuracy_score,
        "Recall": recall_score
    }.get(target_metric, f1_score)

    def objective(trial: optuna.Trial):
        # 1. Предлагаем новые параметры для этой пробы
        trial_params = base_params.copy()
        trial_params['ml_iterations'] = trial.suggest_int('ml_iterations', *param_ranges['ml_iterations'])
        trial_params['ml_depth'] = trial.suggest_int('ml_depth', *param_ranges['ml_depth'])
        trial_params['ml_learning_rate'] = trial.suggest_float('ml_learning_rate', *param_ranges['ml_learning_rate'])
        trial_params['ml_prints_window'] = trial.suggest_int('ml_prints_window', *param_ranges['ml_prints_window'])
        trial_params['ml_labeling_timeout_candles'] = trial.suggest_int('ml_labeling_timeout_candles', *param_ranges['ml_labeling_timeout_candles'])

        # 2. Генерируем признаки и размечаем данные с новыми параметрами
        # Это необходимо, так как 'ml_prints_window' влияет на признаки
        try:
            df_with_features = generate_features(df_with_indicators, trial_params)
            X, y = label_all_signals(df_with_features, signal_indices, trial_params)

            if X.empty or y.nunique() < 2:
                raise optuna.exceptions.TrialPruned("Недостаточно данных для обучения после разметки.")

            # 3. Разделяем на train/test (shuffle=False для сохранения временного порядка)
            test_size = st.session_state.get("ml_test_size_pct", 30) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            if X_train.empty or X_test.empty:
                raise optuna.exceptions.TrialPruned("Недостаточно данных в train или test выборке.")

            # 4. Обучаем модель на train-выборке
            model_bundle = train_ml_model(X_train, y_train, trial_params)

            # 5. Оцениваем на test-выборке
            model = model_bundle['model']
            scaler = model_bundle['scaler']
            feature_names = model_bundle['feature_names']
            numerical_features = model_bundle['numerical_features']

            X_test_scaled = X_test.copy()
            num_features_in_df = [f for f in numerical_features if f in X_test_scaled.columns]
            if scaler and num_features_in_df:
                X_test_scaled[num_features_in_df] = scaler.transform(X_test_scaled[num_features_in_df])

            y_pred = model.predict(X_test_scaled[feature_names])

            # 6. Считаем и возвращаем целевую метрику
            score = metric_func(y_test, y_pred, zero_division=0)

            # Сохраняем все метрики для анализа
            trial.set_user_attr("F1-Score", f1_score(y_test, y_pred, zero_division=0))
            trial.set_user_attr("Precision", precision_score(y_test, y_pred, zero_division=0))
            trial.set_user_attr("Accuracy", accuracy_score(y_test, y_pred))
            trial.set_user_attr("Recall", recall_score(y_test, y_pred, zero_division=0))

            return score

        except Exception as e:
            # В случае ошибки "провалим" пробу
            raise optuna.exceptions.TrialPruned(f"Ошибка в пробе: {e}")

    # Запускаем оптимизацию
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    # Собираем результаты
    best_trial = study.best_trial
    
    # Переобучаем лучшую модель на полных данных для получения бандла
    best_params_final = base_params.copy()
    best_params_final.update(best_trial.params)
    df_features_final = generate_features(df_with_indicators, best_params_final)
    X_final, y_final = label_all_signals(df_features_final, signal_indices, best_params_final)
    X_train_final, _, y_train_final, _ = train_test_split(X_final, y_final, test_size=0.3, shuffle=False)
    best_model_bundle = train_ml_model(X_train_final, y_train_final, best_params_final)

    # Собираем топ-10 проб
    top_10_trials = []
    for t in sorted(study.trials, key=lambda x: x.value or -1, reverse=True)[:10]:
        if t.value is not None:
            trial_data = {"value": t.value, **t.params, **t.user_attrs}
            top_10_trials.append(trial_data)

    results = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_model_bundle": best_model_bundle,
        "top_10_trials": top_10_trials,
        "study": study
    }
    st.session_state['best_ml_results'] = results
    return results