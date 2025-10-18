import numpy as np
import pandas as pd
import hashlib
from sklearn.preprocessing import StandardScaler

from signal_generator import generate_signals, find_future_outcomes

import logging
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

CLASSIFIER_MAPPING = {}
if CATBOOST_AVAILABLE:
    CLASSIFIER_MAPPING["CatBoost"] = CatBoostClassifier

# --- Глобальный кэш для ускорения повторных вычислений в Optuna/WFO ---
# Ключ - хэш от параметров, Значение - (DataFrame с признаками, целевая переменная)
ML_DATA_CACHE = {}

def train_ml_model(df: pd.DataFrame, params: dict, signal_indices: list):
    """
    Полный цикл обучения ML-модели: генерация признаков, кэширование, обучение.

    Args:
        df (pd.DataFrame): DataFrame с рыночными данными.
        params (dict): Словарь с параметрами стратегии и модели.
        signal_indices (list): Индексы базовых сигналов для обучения.

    Returns:
        dict: Словарь с результатами обучения:
              'model', 'scaler', 'feature_df', 'feature_importances', 'used_params'.
              В случае неудачи 'model' будет None, а 'failure_reason' будет содержать причину.
    """
    classifier_type = params.get("classifier_type")
    if not (classifier_type and classifier_type in CLASSIFIER_MAPPING):
        return {
            'model': None, 'scaler': None, 'feature_df': None, 
            'feature_importances': None, 'used_params': set(),
            'failure_reason': "Тип классификатора не указан или не поддерживается."
        }

    used_params = set()

    # --- Оптимизация: Кэширование генерации признаков и целей ---
    # 1. Создаем уникальный ключ для кэширования
    ml_param_keys = [
        'vol_period', 'vol_pctl', 'range_period', 'rng_pctl', 'natr_period', 'natr_min',
        'lookback_period', 'min_growth_pct', 'stop_loss_pct', 'take_profit_pct', 'bracket_timeout_candles',
        'hldir_window', 'hldir_offset', 'prints_analysis_period', 'prints_threshold_ratio',
        'm_analysis_period', 'm_threshold_ratio'
    ]
    relevant_params = {k: params.get(k) for k in ml_param_keys if k in params}
    params_str = str(sorted(relevant_params.items()))
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    cache_key = f"{df_hash}_{params_str}"

    # 2. Проверяем кэш
    if cache_key in ML_DATA_CACHE:
        feature_df, target, ml_feature_gen_used_params = ML_DATA_CACHE[cache_key]
        used_params.update(ml_feature_gen_used_params)
    else:
        # Если в кэше нет, выполняем дорогостоящие вычисления
        # 1. Подготовка признаков (X)
        _, all_indicators, ml_feature_gen_used_params = generate_signals(df, params, base_signal_only=False)
        used_params.update(ml_feature_gen_used_params)
        
        feature_df = pd.DataFrame(all_indicators)
        feature_df['price_range'] = df['high'] - df['low']
        feature_df['volume'] = df['volume']
        feature_df.fillna(0, inplace=True)
        feature_df.replace([np.inf, -np.inf], 0, inplace=True)

        # 2. Определяем целевую переменную (y)
        profit_target_for_labeling = params.get("take_profit_pct", 4.0)
        loss_limit_for_labeling = params.get("stop_loss_pct", 2.0)
        # Горизонт разметки должен быть достаточно большим
        look_forward_for_labeling = params.get("bracket_timeout_candles", 5) * 5 
        bracket_offset_for_labeling = params.get("bracket_offset_pct", 0.5)

        promising_long, _ = find_future_outcomes(
            df['close'].values, df['high'].values, df['low'].values, df['open'].values,
            look_forward_period=look_forward_for_labeling,
            profit_target_ratio=profit_target_for_labeling / 100,
            loss_limit_ratio=loss_limit_for_labeling / 100,
            bracket_offset_ratio=bracket_offset_for_labeling / 100,
            bracket_timeout_candles=params.get("bracket_timeout_candles", 5)
        )
        target = np.zeros(len(df))
        target[promising_long] = 1

        # 3. Сохраняем результат в кэш (без scaler)
        ML_DATA_CACHE[cache_key] = (feature_df, target, ml_feature_gen_used_params)

    # --- Подготовка данных и обучение модели ---
    used_params.add("classifier_type")
    X = feature_df.loc[signal_indices]
    y = target[signal_indices]

    model = None
    feature_importances = None
    failure_reason = None

    if len(X) <= 10:
        logging.warning(f"[ML Handler] Недостаточно сигналов для обучения (требуется > 10, найдено: {len(X)}).")
        failure_reason = f"Недостаточно сигналов для обучения (требуется > 10, найдено: {len(X)})."
    elif len(np.unique(y)) <= 1:
        unique_class = np.unique(y)[0] if len(np.unique(y)) > 0 else 'N/A'
        logging.warning(f"[ML Handler] Все {len(y)} сигналов относятся к одному классу ({unique_class}). Модель не будет обучаться.")
        failure_reason = (f"Все сигналы ({len(y)}) относятся к одному классу. "
                          "Модели не на чем учиться. Попробуйте изменить параметры разметки "
                          "(SL/TP, горизонт скрининга) или базовые фильтры стратегии.")
    else:
        # --- ИСПРАВЛЕНИЕ: Scaler создается и обучается здесь, на актуальных данных ---
        scaler = StandardScaler()
        logging.info(f"[ML Handler] Обучение Scaler и модели '{classifier_type}' на {len(X)} сигналах.")
        X_scaled = scaler.fit_transform(X) # Используем fit_transform

        logging.info(f"[ML Handler] Запуск обучения модели '{classifier_type}' на {len(X)} сигналах.")
        model_class = CLASSIFIER_MAPPING[classifier_type] # type: ignore
        model_init_params = {k.replace(classifier_type.lower() + '_', ''): v for k, v in params.items() if k.startswith(classifier_type.lower() + '_')}
        used_params.update([k for k in params.keys() if k.startswith(classifier_type.lower() + '_')])

        try:
            model = model_class(**model_init_params, random_state=42, verbose=0)
            model.fit(X_scaled, y)
            if hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
                feature_names = X.columns.tolist()
                feature_importances = dict(zip(feature_names, importances))
                logging.info("[ML Handler] Модель успешно обучена.")
        except Exception as e: # type: ignore
            failure_reason = f"Внутренняя ошибка классификатора '{classifier_type}': {e}"            
            model = None
            feature_importances = None

    return {
        'model': model, 'scaler': scaler, 'feature_df': feature_df,
        'feature_importances': feature_importances, 'used_params': used_params,
        'failure_reason': failure_reason
    }