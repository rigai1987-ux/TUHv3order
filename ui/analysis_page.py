import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import streamlit as st
# Импортируем Numba-функции из симулятора для разметки
from trading_simulator import find_bracket_entry, find_first_exit

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
    # Получаем только один параметр, нужный для проверки границ данных
    labeling_timeout = params.get('ml_labeling_timeout_candles', 20)

    labels = []
    valid_signal_indices = []

    # Извлекаем массивы numpy для Numba
    open_prices = df_with_features['open'].values
    high_prices = df_with_features['high'].values
    low_prices = df_with_features['low'].values
    close_prices = df_with_features['close'].values
    hldir_values = df_with_features['HLdir'].values if 'HLdir' in df_with_features.columns else np.zeros_like(open_prices)

    for signal_idx in signal_indices:
        # Пропускаем сигналы слишком близко к концу данных
        # Нужен запас свечей для таймаута входа и таймаута разметки
        if signal_idx + params.get("bracket_timeout_candles", 5) + labeling_timeout >= len(df_with_features):
            continue

        # --- ИСПОЛЬЗУЕМ УНИФИЦИРОВАННУЮ ФУНКЦИЮ ---
        outcome = get_trade_outcome(
            signal_idx, params, open_prices, high_prices, low_prices, close_prices, hldir_values
        )

        # Если outcome == -1 (входа не было), мы просто пропускаем этот сигнал,
        # так как он не несет информации для обучения.
        if outcome == -1:
            continue
        
        labels.append(outcome) # outcome здесь может быть только 0 или 1
        valid_signal_indices.append(signal_idx)

    if not valid_signal_indices:
        return pd.DataFrame(), pd.Series(dtype='int')

    # --- ИСПРАВЛЕНИЕ: Используем .iloc для доступа по целочисленным позициям ---
    # valid_signal_indices - это список позиций (row numbers), а не меток индекса.
    # .iloc гарантирует, что мы выберем правильные строки по их порядковому номеру.
    X = df_with_features.iloc[valid_signal_indices]
    y = pd.Series(labels, index=valid_signal_indices)

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
                                  df_copy['short_prints'].rolling(window=prints_window).sum())
    df_copy['market_strength'] = (df_copy['LongM'].rolling(window=prints_window).sum() - 
                                  df_copy['ShortM'].rolling(window=prints_window).sum())

    # 2. Признаки на основе свечей
    df_copy['hldir_strength'] = df_copy['HLdir'].rolling(window=prints_window).mean()
    body_size = abs(df_copy['close'] - df_copy['open'])
    candle_range = df_copy['high'] - df_copy['low']
    df_copy['relative_body_size'] = body_size / candle_range.replace(0, np.nan)
    df_copy['upper_wick_ratio'] = (df_copy['high'] - np.maximum(df_copy['open'], df_copy['close'])) / candle_range.replace(0, np.nan)

    # 3. Временные признаки
    if 'datetime' in df_copy.columns:
        df_copy['hour_of_day'] = df_copy['datetime'].dt.hour

    return df_copy.fillna(0) # Заполняем возможные NaN нулями

def train_ml_model(X: pd.DataFrame, y: pd.Series, ml_params: dict):
    """
    Обучает модель CatBoostClassifier на предоставленных данных.
    
    Args:
        X (pd.DataFrame): DataFrame с признаками.
        y (pd.Series): Series с метками (0 или 1).
        ml_params (dict): Словарь с гиперпараметрами для CatBoost.

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
    X_train_scaled = X_train.copy()
    if numerical_features:
        # Масштабируем только числовые признаки
        X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])

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

    # Убираем st.spinner, так как эта функция может вызываться из фонового потока Optuna,
    # что приводит к ошибке "missing ScriptRunContext".
    model.fit(X_train_scaled, y)

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

    # Получаем важность признаков
    feature_importances = pd.DataFrame({
        'feature': available_features,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    # --- НОВОЕ: Добавляем столбец с описаниями ---
    feature_importances['description'] = feature_importances['feature'].map(feature_descriptions)
    
    # Возвращаем "бандл" с моделью и всем необходимым для предсказаний
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": available_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features, # Добавляем информацию о категориальных признаках
        "feature_importances": feature_importances,
    }
    
    return model_bundle

def show_analysis_page():
    """
    Отображает страницу "Анализ сигналов".
    (Placeholder - функционал будет добавлен позже)
    """
    st.header("Страница 'Анализ сигналов'")
    st.info("Функционал для анализа сигналов находится в разработке.")
    
    # Возвращаем "бандл" с моделью и всем необходимым для предсказаний
    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": available_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features, # Добавляем информацию о категориальных признаках
        "feature_importances": feature_importances,
    }
    
    return model_bundle