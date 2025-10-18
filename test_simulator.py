import unittest
import pandas as pd
import numpy as np
from trading_simulator import run_trading_simulation, TradingSimulator
from signal_generator import generate_signals

class TestTradingSimulator(unittest.TestCase):

    def setUp(self):
        """
        Подготовка тестовых данных перед каждым тестом.
        """
        # --- Настраиваем параметры симуляции ---
        # Этот блок должен быть в начале, чтобы self.params был доступен остальному коду.
        self.params = {
            # Параметры для signal_generator
            "vol_period": 20, "vol_pctl": 90,
            "range_period": 20, "rng_pctl": 90,
            "natr_period": 10, "natr_min": 0.1,
            "lookback_period": 20, "min_growth_pct": 1.0,
            "prints_analysis_period": 2, "prints_threshold_ratio": 1.5,
            "m_analysis_period": 2, "m_threshold_ratio": 1.5,
            "hldir_window": 10, "hldir_offset": 0,
            
            # Параметры для симулятора
            "position_size": 1000.0, "commission": 0.1,
            "stop_loss_pct": 2.0, "take_profit_pct": 4.0,
            
        }

        # Создаем DataFrame с 40 свечами
        num_rows = 40
        self.df = pd.DataFrame({
            'time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='min')),
            'open': np.full(num_rows, 100.0),
            'high': np.full(num_rows, 102.0),
            'low': np.full(num_rows, 98.0),
            'close': np.full(num_rows, 101.0),
            'volume': np.full(num_rows, 1000.0),
            'long_prints': np.full(num_rows, 10.0),
            'short_prints': np.full(num_rows, 10.0),
            'HLdir': np.full(num_rows, 0.5), # Нейтральное значение
            'LongM': np.full(num_rows, 10.0),
            'ShortM': np.full(num_rows, 10.0),
        })

        # --- Настраиваем "идеальный" сигнал на свече с индексом 25 ---
        self.signal_idx = 25

        # 1. Условия для generate_signals (основные фильтры)
        #    - Низкий объем и узкий диапазон для срабатывания фильтров
        self.df.loc[self.signal_idx, 'volume'] = 10 
        self.df.loc[self.signal_idx, 'high'] = 101.1
        self.df.loc[self.signal_idx, 'low'] = 100.9
        #    - Рост цены для прохождения фильтра роста
        self.df.loc[self.signal_idx - 20, 'close'] = 90 
        #    - Высокий NATR (за счет увеличения диапазона на предыдущих свечах)
        self.df.loc[20:24, 'high'] = 105
        self.df.loc[20:24, 'low'] = 95

        # 2. Условия для стратегии направления "Принты и HLdir" (для LONG)
        prints_period = self.params.get("prints_analysis_period", 2)
        m_period = self.params.get("m_analysis_period", 2)
        hldir_period = self.params.get("hldir_window", 10)

        #    - prints_long должен быть True: long_prints > short_prints * ratio
        self.df.loc[self.signal_idx - prints_period + 1 : self.signal_idx, 'long_prints'] = 100
        #    - hldir_long должен быть True: средний HLdir > 0.5
        self.df.loc[self.signal_idx - hldir_period + 1 : self.signal_idx + 1, 'HLdir'] = 0.8
        #    - m_long должен быть True (на случай если фильтр M-Ratio включен)
        self.df.loc[self.signal_idx - m_period + 1 : self.signal_idx, 'LongM'] = 100

    def test_prints_and_hldir_long_signal(self):
        """
        Тестирует, что симулятор корректно открывает LONG сделку
        по стратегии "Принты и HLdir".
        """
        # Запускаем симуляцию
        results = run_trading_simulation(self.df, self.params)

        # --- Проверяем результаты ---
        
        # 1. Проверяем, что была открыта ровно одна сделка
        self.assertEqual(results['total_trades'], 1, 
                         f"Ожидалась 1 сделка, но получено: {results['total_trades']}")

        # 2. Получаем информацию о сделке
        trade = results['trades'][0]

        # 3. Проверяем, что сделка была открыта на следующей свече после сигнала
        self.assertEqual(trade['entry_idx'], self.signal_idx + 1,
                         "Сделка открыта на неправильной свече.")

        # 4. Проверяем, что направление сделки - LONG
        self.assertEqual(trade['direction'], 'long',
                         "Направление сделки должно быть 'long'.")
        
        print("\nТест 'test_prints_and_hldir_long_signal' успешно пройден!")
        print(f"Сделка открыта по цене {trade['entry_price']} на свече {trade['entry_idx']}.")
        print(f"Причина выхода: {trade['exit_reason']} по цене {trade['exit_price']}.")
        print(f"PnL: ${trade['pnl']:.2f}")

    def test_avoids_infinite_loop_on_ignored_signal(self):
        """
        Тестирует, что симулятор не зацикливается на сигнале, который
        не проходит проверку направления (direction_strategy_func возвращает None).

        Этот тест выявил бы ошибку в старой версии, где отсутствовал
        блок `else: current_idx = analysis_idx`, что приводило к бесконечному циклу.
        """
        # --- 1. Настройка данных с двумя сигналами ---
        # Первый сигнал (ложный) на свече 25
        ignored_signal_idx = 25
        # Второй сигнал (настоящий) на свече 30
        valid_signal_idx = 30

        # Настраиваем оба индекса так, чтобы они прошли основные фильтры generate_signals
        for idx in [ignored_signal_idx, valid_signal_idx]:
            self.df.loc[idx, 'volume'] = 10
            self.df.loc[idx, 'high'] = 101.1
            self.df.loc[idx, 'low'] = 100.9
            self.df.loc[idx - self.params['lookback_period'], 'close'] = 90

        # --- 2. Настройка условий для каждого сигнала ---

        # Для ПЕРВОГО сигнала (ignored_signal_idx=25) создаем условия,
        # при которых direction_strategy_func вернет None.
        # Например, prints_dir будет 'long', а hldir_dir будет 'short'.
        self.df.loc[ignored_signal_idx, 'long_prints'] = 100 # prints -> long
        self.df.loc[ignored_signal_idx, 'HLdir'] = 0.2       # hldir -> short

        # Для ВТОРОГО сигнала (valid_signal_idx=30) создаем "идеальные" условия для LONG.
        prints_period = self.params.get("prints_analysis_period", 2)
        m_period = self.params.get("m_analysis_period", 2)
        hldir_period = self.params.get("hldir_window", 10)

        self.df.loc[valid_signal_idx - prints_period + 1 : valid_signal_idx, 'long_prints'] = 100
        self.df.loc[valid_signal_idx - hldir_period + 1 : valid_signal_idx + 1, 'HLdir'] = 0.8
        self.df.loc[valid_signal_idx - m_period + 1 : valid_signal_idx, 'LongM'] = 100

        # --- 3. Запуск симуляции ---
        # В старой версии кода этот вызов привел бы к бесконечному циклу и зависанию теста.
        results = run_trading_simulation(self.df, self.params)

        # --- 4. Проверка результатов ---
        # Проверяем, что симулятор проигнорировал первый сигнал и обработал второй.

        # Ожидаем ровно одну сделку (по второму сигналу)
        self.assertEqual(results['total_trades'], 1,
                         f"Ожидалась 1 сделка, но получено: {results['total_trades']}. "
                         "Симулятор мог застрять или неправильно обработать сигналы.")

        # Проверяем, что сделка открыта по второму, правильному сигналу
        trade = results['trades'][0]
        self.assertEqual(trade['entry_idx'], valid_signal_idx + 1,
                         "Сделка открыта не по второму (валидному) сигналу.")

        # Проверяем, что направление сделки - LONG
        self.assertEqual(trade['direction'], 'long',
                         "Направление сделки должно быть 'long'.")

        print("\nТест 'test_avoids_infinite_loop_on_ignored_signal' успешно пройден!")
        print("Симулятор корректно проигнорировал ложный сигнал и обработал следующий.")

    def test_regression_full_simulation(self):
        """
        Регрессионный тест для проверки неизменности результатов полной симуляции.
        Этот тест использует более крупный и разнообразный набор данных и проверяет
        ключевые итоговые метрики. Если вы вносите изменения в логику расчетов,
        этот тест должен упасть, сигнализируя о необходимости проверить изменения.
        """
        # --- 1. Подготовка данных и параметров ---
        # Создаем более крупный и реалистичный DataFrame
        num_rows = 500
        np.random.seed(42) # для воспроизводимости
        base_price = 100
        price_changes = np.random.randn(num_rows).cumsum() * 0.1
        close = base_price + price_changes
        high = close + np.random.uniform(0, 1, num_rows)
        low = close - np.random.uniform(0, 1, num_rows)
        open_prices = close - np.random.uniform(-0.2, 0.2, num_rows)

        df = pd.DataFrame({
            'time': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_rows, freq='min')),
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(100, 2000, num_rows),
            'long_prints': np.random.uniform(5, 15, num_rows),
            'short_prints': np.random.uniform(5, 15, num_rows),
            'HLdir': np.random.uniform(0, 1, num_rows),
            'LongM': np.random.uniform(5, 15, num_rows),
            'ShortM': np.random.uniform(5, 15, num_rows),
        })

        # --- Параметры из вашего профиля 'HEMIfull.json' ---
        params = {
            "position_size": 100.0,
            "commission": 0.1,
            "hldir_offset": 5,
            "vol_period": 10,
            "vol_pctl": 1.0,
            "range_period": 4,
            "rng_pctl": 1.0,
            "m_analysis_period": 2,
            "m_threshold_ratio": 1.51,
            "natr_period": 56,
            "natr_min": 0.31,
            "lookback_period": 80,
            "min_growth_pct": 1.0,
            "prints_analysis_period": 2,
            "prints_threshold_ratio": 1.99,
            "hldir_window": 10,
            "stop_loss_pct": 2.06,
            "take_profit_pct": 5.94,
            "use_climax_exit": True,
            "climax_exit_window": 8,
            "climax_exit_threshold": 14.95,
        }

        # --- 2. Запуск симуляции ---
        results = run_trading_simulation(df, params)

        # --- 3. Определение и проверка эталонных значений ---
        # ВАЖНО: Эти значения получены при первом успешном прогоне этого теста.
        # Если вы меняете логику и уверены, что новые расчеты верны,
        # обновите эти значения на новые, полученные из вывода упавшего теста.
        expected_total_trades = 0
        expected_winning_trades = 0
        expected_win_rate = 0.0
        expected_total_pnl = 0.0 # Округлено до 2 знаков

        print(f"\n--- Результаты регрессионного теста ---")
        print(f"Получено сделок: {results['total_trades']}, Ожидалось: {expected_total_trades}")
        print(f"Итоговый PnL: {results['total_pnl']:.2f}, Ожидался: {expected_total_pnl}")

        self.assertEqual(results['total_trades'], expected_total_trades, "Количество сделок не совпадает с эталоном.")
        self.assertEqual(results['winning_trades'], expected_winning_trades, "Количество прибыльных сделок не совпадает с эталоном.")
        self.assertAlmostEqual(results['win_rate'], expected_win_rate, places=2, msg="Win Rate не совпадает с эталоном.")
        self.assertAlmostEqual(results['total_pnl'], expected_total_pnl, places=2, msg="Итоговый PnL не совпадает с эталоном.")
        print("--- Регрессионный тест успешно пройден! ---")

# Эта часть позволяет запускать тест напрямую из командной строки
if __name__ == '__main__':
    unittest.main()
