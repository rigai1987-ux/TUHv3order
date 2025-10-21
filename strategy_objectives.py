import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from trading_simulator import run_trading_simulation
# Импортируем необходимые функции для ML
from signal_generator import generate_signals
from ml_model_handler import label_all_signals, generate_features, train_ml_model

def trading_strategy_objective_sqn(
    data: pd.DataFrame, 
    params: Dict[str, Any]
) -> float:
    """
    Целевая функция, основанная на метрике SQN (System Quality Number) Вана Тарпа.
    SQN = (Mean(PnL) / StdDev(PnL)) * Sqrt(NumTrades)

    Args:
        data: DataFrame с рыночными данными.
        params: Словарь с параметрами стратегии.

    Returns:
        Значение SQN или штраф.
    """
    # Запускаем симуляцию в обычном (не агрессивном) режиме для реалистичной оценки
    results = run_trading_simulation(data, params)

    # Получаем порог из параметров, по умолчанию 25
    min_trades_threshold = params.get('min_trades_threshold', 25)

    total_trades = results.get('total_trades', 0)
    total_pnl = results.get('total_pnl', 0.0)
    profit_factor = results.get('profit_factor', 0.0)
    pnl_history = results.get('pnl_history', [])

    # 1. Жесткий штраф за малое количество сделок
    if total_trades < min_trades_threshold:
        # Сильный штраф, который немного уменьшается по мере приближения к порогу
        return -100.0 + total_trades, results # type: ignore

    # 2. Жесткий штраф за убыточность или отсутствие сделок
    if total_pnl <= 0 or profit_factor < 1.0:
        # Возвращаем отрицательное значение, чтобы отсечь эти результаты.
        # Штраф зависит от profit_factor, чтобы оптимизатор мог двигаться в правильном направлении.        
        return (profit_factor - 2.0), results

    # 3. Расчет SQN (System Quality Number)
    if len(pnl_history) < 2:
        return -50.0, results # Недостаточно данных для расчета стандартного отклонения

    pnl_mean = np.mean(pnl_history)
    pnl_std = np.std(pnl_history, ddof=1) # Используем ddof=1 для выборки

    if pnl_std == 0:
        # Если все PnL одинаковы и положительны, это идеальная, но маловероятная ситуация.
        # Даем высокое значение, но не бесконечность.
        return 10.0 if pnl_mean > 0 else 0.0
    
    sqn = (pnl_mean / pnl_std) * np.sqrt(total_trades)

    # Ограничиваем итоговое значение для стабильности
    clipped_score = np.clip(sqn, -10.0, 10.0)
    
    return clipped_score, results

def trading_strategy_objective_sortino(
    data: pd.DataFrame, 
    params: Dict[str, Any]
) -> float:
    """
    Целевая функция, основанная на Коэффициенте Сортино.
    Sortino Ratio = (Mean(PnL) - RiskFreeRate) / DownsideDeviation
    (RiskFreeRate принимаем за 0)

    Args:
        data: DataFrame с рыночными данными.
        params: Словарь с параметрами стратегии.
        min_trades_threshold: Минимальное количество сделок.

    Returns:
        Значение Коэффициента Сортино или штраф.
    """
    results = run_trading_simulation(data, params)
    pnl_history = results.get('pnl_history', [])

    # Получаем порог из параметров, по умолчанию 25
    min_trades_threshold = params.get('min_trades_threshold', 25)

    if len(pnl_history) < min_trades_threshold:
        return -100.0 + len(pnl_history), results

    pnl_mean = np.mean(pnl_history)
    negative_returns = [r for r in pnl_history if r < 0]
    
    if len(negative_returns) < 2: # Не можем посчитать стандартное отклонение убытков
        return pnl_mean if pnl_mean > 0 else 0.0 # Если убытков нет, возвращаем среднюю прибыль

    downside_std = np.std(negative_returns, ddof=1)
    if downside_std == 0:
        return 10.0 # Если все убытки одинаковы, даем высокое значение

    sortino_ratio = pnl_mean / downside_std
    return np.clip(sortino_ratio, -10.0, 10.0), results

def trading_strategy_multi_objective(
    data: pd.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    """
    Многоцелевая функция, оптимизирующая:
    - Максимизирует SQN
    - Минимизирует Max Drawdown
    - Максимизирует эффективность сигналов (отношение сделок к сигналам)

    Args:
        data: DataFrame с рыночными данными.
        params: Словарь с параметрами стратегии.

    Returns:
        Кортеж, содержащий ( (SQN, -MaxDrawdown, SignalEfficiency), results ).
    """
    results = run_trading_simulation(data, params)
    pnl_history = results.get('pnl_history', [])
    total_trades = results.get('total_trades', 0)
    max_drawdown = results.get('max_drawdown', 1.0) # По умолчанию 100% просадка
    total_signals = results.get('total_signals', 0)

    # Получаем порог из параметров, по умолчанию 25
    min_trades_threshold = params.get('min_trades_threshold', 25)

    # Штраф за малое количество сделок
    if total_trades < min_trades_threshold:
        return (-100.0, -1.0, -1.0), results # Худшие значения для всех целей

    # Расчет SQN
    if len(pnl_history) >= 2:
        pnl_mean = np.mean(pnl_history)
        pnl_std = np.std(pnl_history, ddof=1)
        if pnl_std > 0:
            sqn = (pnl_mean / pnl_std) * np.sqrt(total_trades)
        else:
            sqn = 10.0 if pnl_mean > 0 else 0.0 # Согласовано с trading_strategy_objective_sqn
    else:
        sqn = -50.0 # Недостаточно данных

    # Расчет эффективности сигналов (Signal Efficiency)
    # Цель: минимизировать количество сигналов и максимизировать отношение сделок к сигналам.
    # Формула: (trades / signals) - (signals / N), где N - константа, например, 10000
    # Это поощряет высокий % исполненных сигналов и штрафует за избыточное их количество.
    if total_signals > 0:
        signal_efficiency_ratio = total_trades / total_signals
        # Штраф за большое количество сигналов. Нормализуем на 10000, чтобы штраф был в разумных пределах.
        signal_count_penalty = total_signals / 10000.0
        signal_efficiency = signal_efficiency_ratio - signal_count_penalty
    else:
        signal_efficiency = -1.0 # Штраф, если сигналов нет вообще

    # Ограничиваем значения
    clipped_sqn = np.clip(sqn, -10.0, 10.0)
    clipped_signal_efficiency = np.clip(signal_efficiency, -1.0, 1.0)
    
    # Возвращаем кортеж для многоцелевой оптимизации (SQN, -MaxDrawdown, SignalEfficiency).
    return (clipped_sqn, -max_drawdown, clipped_signal_efficiency), results

def trading_strategy_objective_hft_score(
    data: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Целевая функция для оценки качества высокочастотной торговли (HFT).
    Она поощряет:
    - Высокий Profit Factor (отношение общей прибыли к общему убытку).
    - Высокий Win Rate (процент прибыльных сделок).
    - Большое количество сделок.
    - Низкую просадку.

    HFT Score = (Profit Factor * Win Rate * log10(Количество сделок)) / (1 + Max Drawdown)

    Args:
        data: DataFrame с рыночными данными.
        params: Словарь с параметрами стратегии.

    Returns:
        Кортеж ( (HFT Score), results ).
    """
    results = run_trading_simulation(data, params)

    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0.0)
    profit_factor = results.get('profit_factor', 0.0)
    max_drawdown = results.get('max_drawdown', 1.0)

    min_trades_threshold = params.get('min_trades_threshold', 25)

    # 1. Штраф за малое количество сделок или убыточность
    if total_trades < min_trades_threshold or profit_factor < 1.0:
        return -100.0 + total_trades, results

    # 2. Расчет HFT Score
    # Используем log10 для сглаживания влияния количества сделок
    log_trades = np.log10(total_trades) if total_trades > 0 else 0

    # Формула, объединяющая метрики. (1 + max_drawdown) в знаменателе для штрафа за просадку.
    hft_score = (profit_factor * win_rate * log_trades) / (1 + max_drawdown)

    # Ограничиваем для стабильности
    clipped_score = np.clip(hft_score, 0, 100.0)

    return clipped_score, results

def trading_strategy_multi_objective_ml(
    data: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    """
    Многоцелевая функция, которая на каждой пробе Optuna:
    1. Генерирует сигналы и обучает ML-модель.
    2. Запускает симуляцию с ML-фильтром.
    3. Возвращает кортеж (SQN, -MaxDrawdown, SignalEfficiency) из этой симуляции.

    Args:
        data: DataFrame с рыночными данными (In-Sample).
        params: Словарь с параметрами стратегии от Optuna.

    Returns:
        Кортеж, содержащий ( (SQN, -MaxDrawdown, SignalEfficiency), results_with_ml_filter ).
    """
    min_trades_threshold = params.get('min_trades_threshold', 25)
    penalty_tuple = (-100.0, -1.0, -1.0) # Худшие значения для всех целей

    try:
        # 1. Генерация сигналов и признаков
        signal_indices, df_with_indicators, _ = generate_signals(data, params, return_indicators=True)
        df_with_features = generate_features(df_with_indicators, params)

        if not signal_indices:
            return (-200.0, -1.0, -1.0), {} # Штраф, если не найдено ни одного сигнала

        # 2. Разметка сигналов для обучения
        X, y = label_all_signals(df_with_features, signal_indices, params)

        if X.empty or y.sum() < 5: # Требуем хотя бы 5 успешных примеров для обучения
            return (-150.0, -1.0, -1.0), {} # Штраф, если недостаточно данных для обучения

        # 3. Обучение ML-модели
        ml_model_bundle = train_ml_model(X, y, params)

        # 4. Запуск симуляции с обученной моделью и включенным фильтром
        simulation_params = params.copy()
        simulation_params['use_ml_filter'] = True
        simulation_params['ml_model_bundle'] = ml_model_bundle

        results_with_ml = run_trading_simulation(data, simulation_params)

        # 5. Расчет и возврат многоцелевых метрик
        # Используем логику из `trading_strategy_multi_objective`
        if results_with_ml.get('total_trades', 0) < min_trades_threshold:
            return penalty_tuple, results_with_ml

        # Передаем результаты симуляции в стандартную многоцелевую функцию
        # для расчета метрик. Это позволяет избежать дублирования кода.
        # Для этого нужно временно "обмануть" функцию, передав ей уже готовые результаты.
        # Однако, проще скопировать логику расчета метрик сюда.
        multi_obj_metrics, _ = trading_strategy_multi_objective(data, simulation_params)
        
        # Возвращаем метрики и результаты симуляции с ML
        return multi_obj_metrics, results_with_ml

    except Exception as e:
        # В случае любой ошибки возвращаем худший результат
        return (-500.0, -1.0, -1.0), {"error": str(e)}


def trading_strategy_objective_ml(
    data: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Целевая функция, которая на каждой пробе Optuna:
    1. Генерирует сигналы с параметрами пробы.
    2. Размечает их.
    3. Обучает ML-модель.
    4. Запускает симуляцию с ML-фильтром.
    5. Возвращает SQN этой симуляции.

    Args:
        data: DataFrame с рыночными данными (In-Sample).
        params: Словарь с параметрами стратегии от Optuna.

    Returns:
        Кортеж ( (SQN), results_with_ml_filter ).
    """
    min_trades_threshold = params.get('min_trades_threshold', 25)

    try:
        # 1. Генерация сигналов и признаков
        signal_indices, df_with_indicators, _ = generate_signals(data, params, return_indicators=True)
        df_with_features = generate_features(df_with_indicators, params)

        if not signal_indices:
            return -200.0, {} # Штраф, если не найдено ни одного сигнала

        # 2. Разметка сигналов для обучения
        X, y = label_all_signals(df_with_features, signal_indices, params)

        if X.empty or y.sum() < 5: # Требуем хотя бы 5 успешных примеров для обучения
            return -150.0, {} # Штраф, если недостаточно данных для обучения

        # 3. Обучение ML-модели
        # Собираем ML-гиперпараметры из `params`. Если их нет, используются значения по умолчанию.
        ml_model_bundle = train_ml_model(X, y, params)

        # 4. Запуск симуляции с обученной моделью и включенным фильтром
        simulation_params = params.copy()
        simulation_params['use_ml_filter'] = True
        simulation_params['ml_model_bundle'] = ml_model_bundle

        results_with_ml = run_trading_simulation(data, simulation_params)

        # 5. Расчет и возврат метрики (SQN)
        # Используем логику из `trading_strategy_objective_sqn`
        if results_with_ml.get('total_trades', 0) < min_trades_threshold:
            return -100.0 + results_with_ml.get('total_trades', 0), results_with_ml

        sqn_score, _ = trading_strategy_objective_sqn(data, simulation_params)
        return sqn_score, results_with_ml

    except Exception as e:
        # В случае любой ошибки возвращаем худший результат
        return -500.0, {"error": str(e)}

def trading_strategy_objective_ml_data_quality(
    data: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Целевая функция, оптимизирующая качество данных для обучения ML-модели.
    Идеально подходит для использования внутри WFO, где ML-модель обучается один раз
    на лучших параметрах, найденных этой функцией.

    Оптимизирует компромисс между:
    1.  **Количеством сигналов**: Поощряет "золотую середину" (не слишком много, не слишком мало).
    2.  **Количеством успешных примеров (y=1)**: Гарантирует, что модели есть на чем учиться.
    3.  **Качеством базовой стратегии (SQN)**: Оценивает потенциал сигналов до ML-фильтрации.

    Args:
        data: DataFrame с рыночными данными (In-Sample).
        params: Словарь с параметрами стратегии от Optuna.

    Returns:
        Кортеж ( (Score), results_without_ml_filter ).
    """
    try:
        # 1. Генерация сигналов и признаков
        signal_indices, df_with_indicators, _ = generate_signals(data, params, return_indicators=True)
        df_with_features = generate_features(df_with_indicators, params)

        # 2. Разметка сигналов для оценки качества данных
        X, y = label_all_signals(df_with_features, signal_indices, params)

        num_signals = len(X)
        num_positive_labels = y.sum()

        # Штрафы за недостаточное количество данных
        if num_signals < params.get('min_trades_threshold', 25):
            return -200.0 + num_signals, {"total_signals": num_signals, "positive_labels": num_positive_labels}
        if num_positive_labels < 5:
            return -150.0 + num_positive_labels, {"total_signals": num_signals, "positive_labels": num_positive_labels}

        # 3. Запуск симуляции БЕЗ ML-фильтра для оценки базового SQN
        # Это важно, чтобы оценить "сырое" качество сигналов
        simulation_params = params.copy()
        simulation_params['use_ml_filter'] = False
        sqn_score, results = trading_strategy_objective_sqn(data, simulation_params)

        # 4. Расчет итоговой оценки
        # Цель: найти баланс между SQN и количеством данных для ML
        # Используем логарифм для сглаживания влияния количества сигналов и позитивных меток
        data_quality_score = np.log1p(num_signals) * np.log1p(num_positive_labels)

        # Итоговая метрика = SQN * Качество данных
        # Если SQN отрицательный (убыточная стратегия), он сильно оштрафует итоговый результат
        final_score = sqn_score * data_quality_score

        return np.clip(final_score, -100.0, 100.0), results

    except Exception as e:
        return -500.0, {"error": str(e)}