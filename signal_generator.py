import numpy as np
import pandas as pd
import numba
import math


import time
def calculate_true_range(high, low, close):
    """
    Вычисление истинного диапазона (True Range) с поддержкой GPU.

    Args:
        high: массив значений максимума
        low: массив значений минимума
        close: массив значений закрытия

    Returns:
        массив значений истинного диапазона
    """
    xp = np
    close_prev = xp.roll(close, 1)
    tr = xp.maximum.reduce([high - low, xp.abs(high - close_prev), xp.abs(low - close_prev)])
    tr[0] = high[0] - low[0]
    return tr


def calculate_atr(high, low, close, period):
    """
    Вычисление среднего истинного диапазона (ATR) с использованием MMA и поддержкой GPU.
    """
    tr = calculate_true_range(high, low, close)
    # Использование pandas ewm - эффективный и проверенный способ
    # Он хорошо работает как с NumPy, так и с CuPy-backed Series в будущем.
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_natr(high, low, close, period):
    """
    Вычисление нормализованного среднего истинного диапазона (NATR)
    
    Args:
        high: массив значений максимума
        low: массив значений минимума
        close: массив значений закрытия
        period: период для вычисления NATR
    
    Returns:
        массив значений NATR в процентах
    """
    atr = calculate_atr(high, low, close, period)
    # Используем np.divide для безопасного деления на ноль
    natr = np.divide(atr, close, out=np.full_like(close, np.nan, dtype=float), where=close!=0) * 100
    return natr

def calculate_cagr(initial_balance, final_balance, days):
    """
    Рассчитывает среднегодовую доходность (CAGR).

    Args:
        initial_balance (float): Начальный баланс.
        final_balance (float): Конечный баланс.
        days (int): Количество дней в периоде.

    Returns:
        float: Значение CAGR или 0.0, если расчет невозможен.
    """
    if initial_balance <= 0 or final_balance <= 0 or days <= 0:
        return 0.0
    
    years = days / 365.25
    if years == 0: return 0.0

    return (final_balance / initial_balance) ** (1 / years) - 1

def generate_signals(df, params, return_indicators=False):
    """
    Генерация сигналов на основе параметров стратегии
    
    Args:
        df: DataFrame с рыночными данными (time, open, high, low, close, volume, long_prints, short_prints)
        params: словарь с параметрами стратегии:
            - vol_period: период для анализа объема (по умолчанию 20)
            - vol_pctl: порог объема в процентах (по умолчанию 1.0)
            - range_period: период для анализа диапазона (по умолчанию 20)
            - rng_pctl: порог диапазона в процентах (по умолчанию 1.0)
            - natr_period: период для NATR (по умолчанию 10)
            - natr_min: минимальный порог NATR в процентах (по умолчанию 0.35)
            - lookback_period: период для фильтра роста (по умолчанию 20)
            - min_growth_pct: минимальный порог роста в процентах (по умолчанию 1.0)
        base_signal_only (bool): Если True, возвращает только индексы сигналов. 
                                 Если False, возвращает DataFrame со всеми индикаторами.
    Returns:
        кортеж (индексы сигналов, DataFrame с индикаторами, набор использованных параметров)
    """
    # Проверяем, что DataFrame не пуст
    if df.empty or 'Symbol' not in df.columns:
        return [], None, set()

    # --- Группировка по символу для корректного расчета индикаторов ---
    # Это критически важно, чтобы rolling-операции не "перетекали" между разными инструментами.
    if df['Symbol'].nunique() > 1:
        all_indices = []
        all_used_params = set()
        
        # Если нужно вернуть DataFrame, собираем их
        dfs_with_indicators = []

        # Сохраняем исходный индекс для восстановления после группировки
        for symbol, group_df in df.groupby('Symbol'):
            # Передаем копию, чтобы избежать SettingWithCopyWarning
            indices, df_with_indicators, used_params = generate_signals(group_df.copy(), params, return_indicators)
            
            # Восстанавливаем исходные индексы для сигналов
            # group_df.index содержит оригинальные индексы из исходного df
            original_group_indices = group_df.index
            all_indices.extend(original_group_indices[indices])
            
            if return_indicators and df_with_indicators is not None:
                dfs_with_indicators.append(df_with_indicators)
            
            all_used_params.update(used_params)
        
        final_df = pd.concat(dfs_with_indicators).sort_index() if dfs_with_indicators else None
        return sorted(all_indices), final_df, all_used_params

    used_params = set()
    
    # Извлекаем параметры с значениями по умолчанию и округляем float параметры до 2 знаков после запятой
    vol_period = int(params.get("vol_period", 20))
    vol_pctl = round(params.get("vol_pctl", 1.0), 2) / 100  # Преобразуем в доли и округляем до 2 знаков # type: ignore
    range_period = int(params.get("range_period", 20))
    rng_pctl = round(params.get("rng_pctl", 1.0), 2) / 100  # Преобразуем в доли и округляем до 2 знаков
    natr_period = int(params.get("natr_period", 10))
    natr_min = round(params.get("natr_min", 0.35), 2)  # Уже в процентах, округляем до 2 знаков
    lookback_period = int(params.get("lookback_period", 20))
    min_growth_pct = round(params.get("min_growth_pct", 1.0), 2) / 100 # Преобразуем в доли и округляем до 2 знаков
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Столбец {col} отсутствует в DataFrame")
    
    # --- НОВОЕ: Создаем копию для добавления индикаторов ---
    indicators_df = df.copy()

    # Извлекаем массивы для вычислений
    high_prices = indicators_df['high'].values
    low_prices = indicators_df['low'].values
    close_prices = indicators_df['close'].values
    open_prices = indicators_df['open'].values
    volume = indicators_df['volume'].values

    # Вычисляем дополнительные столбцы
    # Диапазон цены (range)
    price_range = high_prices - low_prices
    
    # Вычисляем NATR
    indicators_df['natr'] = calculate_natr(
        high_prices, low_prices, close_prices,
        natr_period
    )
    
    # Вычисляем процентный рост за lookback_period
    
    # Определяем GPU и CPU функции для вычисления роста
    indicators_df['growth_pct'] = 0.0
    # --- ИСПРАВЛЕНИЕ: Используем .iloc для позиционной индексации ---
    # Это предотвращает ошибку ValueError, когда DataFrame является срезом с не-нулевыми индексами.
    indicators_df.iloc[lookback_period:, indicators_df.columns.get_loc('growth_pct')] = (
        (close_prices[lookback_period:] - close_prices[:-lookback_period]) /
        close_prices[:-lookback_period]
    ) * 100
    
    # Создаем скользящие процентили для объема и диапазона
    # Используем rolling с квантилями для эффективного вычисления процентилей
    # Современная реализация в Pandas достаточно быстра и корректно работает с GIL в многопоточном режиме.
    volume_percentiles = np.full_like(volume, np.nan, dtype=float)
    if vol_period > 0 and len(volume) >= vol_period:
        used_params.update({"vol_period", "vol_pctl"})
        volume_percentiles = pd.Series(volume).rolling(window=vol_period, min_periods=1).quantile(vol_pctl, interpolation='lower').values

    range_percentiles = np.full_like(price_range, np.nan, dtype=float)
    if range_period > 0 and len(price_range) >= range_period:
        used_params.update({"range_period", "rng_pctl"})
        range_percentiles = pd.Series(price_range).rolling(window=range_period, min_periods=1).quantile(rng_pctl, interpolation='lower').values

    
    # Заполняем NaN в начале, которые могут возникнуть из-за скользящего окна
    volume_percentiles = pd.Series(volume_percentiles).ffill().values
    range_percentiles = pd.Series(range_percentiles).ffill().values
    used_params.update({"natr_period", "natr_min"})
    used_params.update({"lookback_period", "min_growth_pct"})

    # Создаем условия для всех строк сразу (векторизованно)
    low_vol_condition = volume <= volume_percentiles
    narrow_rng_condition = price_range <= range_percentiles
    high_natr_condition = indicators_df['natr'].values > natr_min
    growth_condition = indicators_df['growth_pct'].values >= (min_growth_pct * 100) # Сравниваем проценты с процентами
    
    # Объединяем первые 4 условия
    signal_conditions = low_vol_condition & narrow_rng_condition & high_natr_condition & growth_condition
    
    # Устанавливаем в False первые несколько индексов, где условия не могут быть вычислены корректно    
    min_period = max(vol_period, range_period, natr_period, lookback_period)
    signal_conditions[:min_period] = False
    
    # Добавляем дополнительную проверку: если NATR не определен (NaN), не генерируем сигнал
    natr_is_nan = np.isnan(indicators_df['natr'].values)
    signal_conditions = signal_conditions & ~natr_is_nan

    # Возвращаем индексы, где выполнены все условия
    # Возвращаем целочисленные позиции индексов, а не сами значения индекса
    final_indices = np.where(signal_conditions)[0].tolist()
    return final_indices, indicators_df if return_indicators else None, used_params
