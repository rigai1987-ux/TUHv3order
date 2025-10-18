import numpy as np
import pandas as pd
import numba
import math
from app_utils import find_bracket_entry

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

@numba.jit(nopython=True, cache=True)
def find_future_outcomes(
    high_prices,
    low_prices,
    close_prices,
    open_prices,
    hldir_values,
    look_forward_period,
    profit_target_ratio,
    loss_limit_ratio,
    bracket_offset_pct,
    bracket_timeout_candles
):
    n = len(high_prices)
    promising_long = np.zeros(n, dtype=numba.boolean)
    promising_short = np.zeros(n, dtype=numba.boolean)

    for i in range(n - look_forward_period):
        base_price = close_prices[i]
        long_level = base_price * (1 + bracket_offset_pct / 100)
        short_level = base_price * (1 - bracket_offset_pct / 100)

        entry_idx, entry_price, direction = find_bracket_entry(
            start_idx=i + 1,
            timeout=bracket_timeout_candles,
            long_level=long_level,
            short_level=short_level,
            high_prices=high_prices,
            low_prices=low_prices,
            open_prices=open_prices,
            hldir_values=hldir_values,
            close_prices=close_prices
        )

        if entry_idx != -1 and direction != "none":
            if direction == "long":
                take_profit_price = entry_price * (1 + profit_target_ratio)
                stop_loss_price = entry_price * (1 - loss_limit_ratio)

                for j in range(entry_idx, min(entry_idx + look_forward_period, n)):
                    if low_prices[j] <= stop_loss_price:
                        promising_long[i] = False
                        break
                    if high_prices[j] >= take_profit_price:
                        promising_long[i] = True
                        break
            
            elif direction == "short":
                take_profit_price = entry_price * (1 - profit_target_ratio)
                stop_loss_price = entry_price * (1 + loss_limit_ratio)

                for j in range(entry_idx, min(entry_idx + look_forward_period, n)):
                    if high_prices[j] >= stop_loss_price:
                        promising_short[i] = False
                        break
                    if low_prices[j] <= take_profit_price:
                        promising_short[i] = True
                        break

    return promising_long, promising_short

def generate_signals(df, params, base_signal_only=False):
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
            - stop_loss_pct: стоп-лосс в процентах (по умолчанию 2.0)
            - take_profit_pct: тейк-профит в процентах (по умолчанию 4.0)
    
    Args:
        base_signal_only (bool): Если True, генерирует сигналы только на основе базовых фильтров (объем, диапазон, NATR, рост), игнорируя специфичные для стратегий индикаторы.
    Returns:
        кортеж (список индексов сигналов, словарь с индикаторами, набор использованных параметров)
        (final_indices, indicators, used_params)
    """
    # Проверяем, что DataFrame не пуст
    if df.empty or 'Symbol' not in df.columns:
        return [], {}, set()

    # --- Группировка по символу для корректного расчета индикаторов ---
    # Это критически важно, чтобы rolling-операции не "перетекали" между разными инструментами.
    if df['Symbol'].nunique() > 1:
        # Если в DataFrame несколько символов, рекурсивно вызываем функцию для каждой группы
        # и объединяем результаты.
        all_indices = []
        all_indicators = {}
        all_used_params = set()
        for symbol, group_df in df.groupby('Symbol'):
            indices, indicators, used_params = generate_signals(group_df, params, base_signal_only)
            # TODO: Эта часть требует доработки для корректного объединения индикаторов.
            # Пока что для множественных символов возвращаем только индексы.
            all_indices.extend(indices)
            all_used_params.update(used_params)
        # ВНИМАНИЕ: Возврат индикаторов для нескольких символов пока не реализован полностью.
        # Для текущей задачи (подсчет сигналов и симуляция) это не критично.
        return all_indices, {}, all_used_params

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
    prints_analysis_period = int(params.get("prints_analysis_period", 2))
    prints_threshold_ratio = params.get("prints_threshold_ratio", 1.0)
    m_analysis_period = int(params.get("m_analysis_period", 2))
    m_threshold_ratio = params.get("m_threshold_ratio", 1.0)
    hldir_window = int(params.get("hldir_window", 10))
    hldir_offset = int(params.get("hldir_offset", 0))
    hldir_mode = params.get("hldir_mode", "rolling_mean") # 'rolling_mean' или 'candle_compare'
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Столбец {col} отсутствует в DataFrame")
    
    # Извлекаем массивы для вычислений
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    open_prices = df['open'].values
    volume = df['volume'].values

    # Вычисляем дополнительные столбцы
    # Диапазон цены (range)
    price_range = high_prices - low_prices
    
    # Вычисляем NATR
    natr_values = calculate_natr(
        high_prices, low_prices, close_prices,
        natr_period
    )
    
    # Вычисляем процентный рост за lookback_period
    
    # Определяем GPU и CPU функции для вычисления роста
    growth_values = np.zeros(len(close_prices))
    growth_values[lookback_period:] = (
        (close_prices[lookback_period:] - close_prices[:-lookback_period]) /
        close_prices[:-lookback_period]
    )
    
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
    volume_percentiles = pd.Series(volume_percentiles).bfill().values
    range_percentiles = pd.Series(range_percentiles).bfill().values
    used_params.update({"natr_period", "natr_min"})
    used_params.update({"lookback_period", "min_growth_pct"})

    # --- Расчет индикаторов для стратегий ---
    def rolling_sum_numpy(arr, window):
        """
        Надежная реализация скользящей суммы с использованием pandas.
        min_periods=1 гарантирует, что сумма будет считаться даже для неполных окон в начале.
        """
        return pd.Series(arr).rolling(window=window, min_periods=1).sum().values

    # Инициализируем индикаторы как False
    prints_long = np.zeros(len(df), dtype=bool)
    prints_short = np.zeros(len(df), dtype=bool)
    m_long = np.zeros(len(df), dtype=bool)
    m_short = np.zeros(len(df), dtype=bool)

    # Рассчитываем индикаторы только если они нужны для выбранной стратегии
    # base_signal_only=True для "Вилки", поэтому эти блоки не выполнятся
    # Для ML-модели (когда base_signal_only=False) мы всегда рассчитываем эти индикаторы как признаки.
    if not base_signal_only:
        used_params.update({"prints_analysis_period", "prints_threshold_ratio"})
        long_sum = rolling_sum_numpy(df['long_prints'].values, prints_analysis_period)
        short_sum = rolling_sum_numpy(df['short_prints'].values, prints_analysis_period)
        # Prints Ratios
        ratios = np.divide(long_sum, short_sum, where=short_sum > 0, out=np.full_like(long_sum, float('inf'), dtype=float))
        prints_long = ratios > prints_threshold_ratio
        prints_short = ratios < (1 / prints_threshold_ratio)

        # M-Ratio
        used_params.update({"m_analysis_period", "m_threshold_ratio"})
        long_m_sum = rolling_sum_numpy(df['LongM'].values, m_analysis_period)
        short_m_sum = rolling_sum_numpy(df['ShortM'].values, m_analysis_period)
        m_ratios = np.divide(long_m_sum, short_m_sum, where=short_m_sum > 0, out=np.full_like(long_m_sum, float('inf'), dtype=float))
        m_long = m_ratios > m_threshold_ratio
        m_short = m_ratios < (1 / m_threshold_ratio)

    used_params.update({"hldir_window", "hldir_offset"})

    if hldir_mode == 'candle_compare':
        # Новая логика на основе сравнения со свечой
        hldir_values = df['HLdir'].values
        cond1_long = (hldir_values == 1) & (high_prices > open_prices)
        cond1_short = (hldir_values == 1) & (high_prices <= open_prices)
        cond2_long = (hldir_values == 0) & (low_prices >= open_prices)
        cond2_short = (hldir_values == 0) & (low_prices < open_prices)
        hldir_long = cond1_long | cond2_long
        hldir_short = cond1_short | cond2_short
    else: # 'rolling_mean' - логика по умолчанию
        hldir_rolling_mean = pd.Series(df['HLdir']).rolling(window=hldir_window, min_periods=1).mean().shift(hldir_offset).values
        hldir_long = hldir_rolling_mean > 0.5
        hldir_short = hldir_rolling_mean <= 0.5

    # --- Конец расчета индикаторов для стратегий ---

    # Создаем условия для всех строк сразу (векторизованно)
    low_vol_condition = volume <= volume_percentiles
    narrow_rng_condition = price_range <= range_percentiles
    high_natr_condition = natr_values > natr_min
    growth_condition = growth_values >= min_growth_pct
    
    # Объединяем первые 4 условия
    signal_conditions = low_vol_condition & narrow_rng_condition & high_natr_condition & growth_condition
    
    # Устанавливаем в False первые несколько индексов, где условия не могут быть вычислены корректно
    min_period = max(vol_period, range_period, natr_period, lookback_period, prints_analysis_period, m_analysis_period)
    signal_conditions[:min_period] = False
    
    # Добавляем дополнительную проверку: если NATR не определен (NaN), не генерируем сигнал
    natr_is_nan = np.isnan(natr_values)
    signal_conditions = signal_conditions & ~natr_is_nan

    # Собираем все индикаторы в один словарь
    indicators = {
        'prints_long': prints_long, 'prints_short': prints_short,
        'm_long': m_long, 'm_short': m_short,
        'hldir_long': hldir_long, 'hldir_short': hldir_short,
    }

    # Возвращаем индексы, где выполнены все условия
    # Возвращаем целочисленные позиции индексов, а не сами значения индекса
    final_indices = np.where(signal_conditions)[0].tolist()
    return final_indices, indicators, used_params
