import unittest
import pandas as pd
import numpy as np
import streamlit as st
from unittest.mock import patch, MagicMock

# Импортируем тестируемую функцию и её зависимости
from ui.analysis_page import run_forced_ml_analysis
from ml_model_handler import train_ml_model
from signal_generator import generate_signals
from visualizer import plot_ml_decision_boundary

class TestForcedMLAnalysis(unittest.TestCase):

    def setUp(self):
        """Подготовка тестовых данных перед каждым тестом."""
        # Создаем DataFrame с 100 свечами
        num_rows = 100
        self.df = pd.DataFrame({
            'time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='min')),
            'Symbol': ['TEST'] * num_rows,
            'open': np.full(num_rows, 100.0),
            'high': np.random.uniform(100.5, 102.0, size=num_rows),
            'low': np.random.uniform(98.0, 99.5, size=num_rows),
            'close': np.random.uniform(99.5, 100.5, size=num_rows),
            'volume': np.random.uniform(100, 1000, size=num_rows),
            'long_prints': np.random.uniform(5, 15, num_rows),
            'short_prints': np.random.uniform(5, 15, num_rows),
            'HLdir': np.random.choice([0, 1], size=num_rows),
            'LongM': np.random.uniform(5, 15, num_rows),
            'ShortM': np.random.uniform(5, 15, num_rows),
        })

        # Настраиваем параметры, включая ML
        self.params = {
            # Базовые параметры для генерации сигналов
            "vol_period": 10, "vol_pctl": 5.0,
            "range_period": 10, "rng_pctl": 5.0,
            "natr_period": 10, "natr_min": 0.1,
            "lookback_period": 10, "min_growth_pct": -5.0, # Ослабляем фильтры, чтобы получить больше сигналов
            
            # Параметры для признаков ML
            "prints_analysis_period": 5, "prints_threshold_ratio": 1.1,
            "m_analysis_period": 5, "m_threshold_ratio": 1.1,
            "hldir_window": 10, "hldir_offset": 0,

            # Параметры для разметки данных (для обучения)
            "take_profit_pct": 2.0, "stop_loss_pct": 1.0,
            "bracket_timeout_candles": 5,

            # Параметры классификатора
            "classifier_type": "CatBoost",
            "catboost_iterations": 10,
            "catboost_depth": 4,
            "catboost_learning_rate": 0.1,
        }

    @patch('ui.analysis_page.st') # Мокаем Streamlit, чтобы избежать ошибок
    def test_run_forced_ml_analysis_returns_figure(self, mock_st):
        """
        Тестирует, что функция `run_forced_ml_analysis` успешно выполняется
        и возвращает объект фигуры Plotly.
        """
        # Мокаем st.warning, чтобы не выводить предупреждения в консоль во время теста
        mock_st.warning = MagicMock()

        # Запускаем тестируемую функцию
        # В реальном приложении `df` и `params` берутся из session_state,
        # здесь мы передаем их напрямую.
        fig = run_forced_ml_analysis(self.df, self.params)

        # --- Проверяем результат ---
        
        # 1. Проверяем, что функция не вернула None
        self.assertIsNotNone(fig, "Функция не должна возвращать None при успешном выполнении.")

        # 2. Проверяем, что возвращенный объект является фигурой Plotly
        import plotly.graph_objects as go
        self.assertIsInstance(fig, go.Figure, "Функция должна возвращать объект plotly.graph_objects.Figure.")

        print("\nТест 'test_run_forced_ml_analysis_returns_figure' успешно пройден!")

if __name__ == '__main__':
    unittest.main()