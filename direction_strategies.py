"""
Модуль для определения направления торговли.
После рефакторинга осталась только стратегия "Вилка отложенных ордеров",
логика которой реализована непосредственно в симуляторе.
"""
import numpy as np
import pandas as pd

def direction_bracket_placeholder(idx, indicators, params, screening_mode=False, promising_long_mask=None, promising_short_mask=None):
    """Заглушка для стратегии "вилки". Логика реализована в trading_simulator."""
    # Эта функция не должна вызываться напрямую, так как логика "вилки" обрабатывается отдельно.
    return None

# Словарь-регистр для всех стратегий