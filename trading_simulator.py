import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signal_generator import generate_signals, find_future_outcomes # type: ignore
# --- ИЗМЕНЕНИЕ: Импортируем новый обработчик ML ---
from ml_model_handler import train_ml_model

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    # Создаем "пустой" декоратор, если Numba не установлен
    def numba_jit_noop(func):
        return func
    
    class numba:
        jit = numba_jit_noop

    NUMBA_AVAILABLE = False

class TradingSimulator:
    """
    Модуль симуляции сделок
    """
    
    def __init__(self, position_size=100.0, commission=0.1, stop_loss_pct=2.0, take_profit_pct=4.0):
       """
       Инициализация симулятора
        
       Args:
           position_size: размер позиции
           commission: комиссия в процентах
           stop_loss_pct: стоп-лосс в процентах
           take_profit_pct: тейк-профит в процентах
           use_hldir: использовать ли HLdir для определения направления (если доступен)
       """
       self.position_size = position_size
       self.commission = commission / 100  # преобразуем в доли
       self.stop_loss_pct = stop_loss_pct
       self.take_profit_pct = take_profit_pct
       self.trades = []
       self.pnl_history = []
       self.balance_history = []
       self.current_position = None
       self.total_signals = 0
       self.initial_balance = position_size
       self.ml_rejected_signals = []
       # Всегда используем HLdir, если он присутствует в данных
        
    def calculate_stop_loss_take_profit(self, entry_price, direction):
        """
        Расчет стоп-лосса и тейк-профита
        
        Args:
            entry_price: цена входа
            direction: направление позиции ('long' или 'short')
            
        Returns:
            кортеж (stop_loss, take_profit)
        """
        if direction == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.take_profit_pct / 100)
        else:  # short
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.take_profit_pct / 100)
            
        return stop_loss, take_profit
    
    def open_position(self, idx, entry_price, direction, signal_data):
        """
        Открытие позиции
        
        Args:
            idx: индекс свечи
            entry_price: цена входа (цена открытия следующей свечи)
            direction: направление ('long' или 'short')
            signal_data: данные сигнала
        """
        if self.current_position is not None:
            # Если позиция уже открыта, не открываем новую
            return False
            
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, direction)
        
        # Конвертируем значения в стандартные типы Python, чтобы избежать проблем с сериализацией
        self.current_position = {
            'entry_idx': int(idx) if hasattr(idx, 'item') else idx,
            'entry_price': float(entry_price) if hasattr(entry_price, 'item') else entry_price,
            'direction': direction,
            'stop_loss': float(stop_loss) if hasattr(stop_loss, 'item') else stop_loss,
            'take_profit': float(take_profit) if hasattr(take_profit, 'item') else take_profit,
            'exit_idx': None,
            'exit_price': None,
            'pnl': None,
            'signal_data': signal_data
        }
        
        return True
    
    def close_position(self, idx, exit_price, reason):
        """
        Закрытие позиции
        
        Args:
            idx: индекс свечи
            exit_price: цена выхода
            reason: причина закрытия ('stop_loss', 'take_profit', 'signal_reverse', 'end_of_data')
        """
        if self.current_position is None:
            return 0  # Нет позиции для закрытия
            
        # Рассчитываем PNL
        entry_price = self.current_position['entry_price']
        direction = self.current_position['direction']
        
        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) / entry_price
            
        # Учитываем размер позиции
        pnl_amount = self.position_size * pnl
        
        # Учитываем комиссии (вход + выход)
        commission_amount = (self.position_size * self.commission) * 2
            
        pnl_amount -= commission_amount
        
        # Проверяем, что сделка приносит минимальную прибыль после вычета комиссий
        # Если прибыль слишком мала или убыток, считаем, что сделка не состоялась
        if abs(pnl_amount) < (commission_amount * 0.5):  # Порог в половину комиссии
            return 0 # Не учитываем сделку как успешную
        
        # Обновляем позицию
        # Конвертируем значения в стандартные типы Python, чтобы избежать проблем с сериализацией
        self.current_position['exit_idx'] = int(idx)
        self.current_position['exit_price'] = float(exit_price)
        self.current_position['pnl'] = float(pnl_amount)
        self.current_position['exit_reason'] = reason
        
        # Добавляем в историю сделок
        self.trades.append(self.current_position.copy())
        
        # Сохраняем PNL в историю
        self.pnl_history.append(pnl_amount)
        
        # Обновляем историю баланса
        balance_value = self.initial_balance + pnl_amount if len(self.balance_history) == 0 else self.balance_history[-1] + pnl_amount
            
        self.balance_history.append(balance_value)
        
        # Сбрасываем текущую позицию
        self.current_position = None
        
        return pnl_amount

    def simulate_trades(self, df, params, aggressive_mode=False, screening_mode=False):
        """
        Симуляция торговли

        Args:
            df: DataFrame с рыночными данными
            params: параметры стратегии
            aggressive_mode: если True, позволяет открывать новые сделки, не дожидаясь закрытия старых
            screening_mode: если True, торгует только по "перспективным" сигналам (для обучения)

        Returns:
            словарь с результатами симуляции
        """
        # Извлекаем параметры для определения направления
        use_climax_exit = params.get("use_climax_exit", False)
        # Новые параметры для входа по "вилке"
        bracket_offset_pct = params.get("bracket_offset_pct", 0.5)
        bracket_timeout_candles = params.get("bracket_timeout_candles", 5)

        climax_exit_window = params.get("climax_exit_window", 50)
        climax_exit_threshold = params.get("climax_exit_threshold", 3.0)

        # Начинаем собирать использованные параметры
        used_params = {"position_size", "commission", "stop_loss_pct", "take_profit_pct"}


        # Определяем, нужно ли генерировать только базовые сигналы (для "Вилки")
        # Теперь всегда генерируем только базовые сигналы, так как другие стратегии удалены
        base_signal_only = True
        # Генерируем базовые сигналы. Индикаторы для ML будут сгенерированы отдельно, если нужно.
        # signal_indices, indicators, signal_gen_used_params = generate_signals(df, params, base_signal_only=False)
        # Возвращаем base_signal_only=True, т.к. индикаторы для ML будем генерировать отдельно
        signal_indices, indicators, signal_gen_used_params = generate_signals(df, params, base_signal_only=base_signal_only)
        used_params.update(signal_gen_used_params)
        self.total_signals = len(signal_indices)
        
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        xp = np
        hldir_values = df['HLdir'].values
        long_prints_values = df['long_prints'].values
        short_prints_values = df['short_prints'].values

        # 3. Предварительно рассчитываем Climax Score для всех свечей
        long_climax_scores = xp.zeros(len(df))
        short_climax_scores = xp.zeros(len(df))
        if use_climax_exit:
            used_params.update({"use_climax_exit", "climax_exit_window", "climax_exit_threshold"})
            price_range = high_prices - low_prices
            # Используем pandas для удобного расчета скользящих окон
            s_long_prints = pd.Series(df['long_prints'].values)
            s_short_prints = pd.Series(df['short_prints'].values)
            s_range = pd.Series(price_range)

            # Скользящие статистики
            rolling_window = s_long_prints.rolling(window=climax_exit_window, min_periods=1)
            long_prints_ma = rolling_window.mean().values
            long_prints_std = rolling_window.std().values
            rolling_window_short = s_short_prints.rolling(window=climax_exit_window, min_periods=1)
            short_prints_ma = rolling_window_short.mean().values
            short_prints_std = rolling_window_short.std().values
            range_ma = s_range.rolling(window=climax_exit_window, min_periods=1).mean().values

            # Z-score и volatility_factor
            long_prints_z = xp.divide(long_prints_values - long_prints_ma, long_prints_std, where=long_prints_std > 1e-6, out=xp.zeros_like(long_prints_values, dtype=float))
            short_prints_z = xp.divide(short_prints_values - short_prints_ma, short_prints_std, where=short_prints_std > 1e-6, out=xp.zeros_like(short_prints_values, dtype=float))
            volatility_factor = xp.divide(price_range, range_ma, where=range_ma > 1e-6, out=xp.zeros_like(price_range, dtype=float))
            long_climax_scores = long_prints_z * volatility_factor
            short_climax_scores = short_prints_z * volatility_factor

        # 2. Итерируемся по сигналам, а не по всем свечам
        entry_indices = sorted(list(signal_indices))
        if not entry_indices: # Если нет сигналов, выходим
            return self.calculate_metrics()

        # --- ЭТАП ПОДГОТОВКИ И ОБУЧЕНИЯ МОДЕЛИ (если используется) ---
        model = None
        scaler = None
        feature_df = None
        feature_importances = None

        # --- ИЗМЕНЕНИЕ: Вызываем новый обработчик ML ---
        # Вся логика обучения, кэширования и генерации признаков теперь инкапсулирована.
        if params.get("classifier_type"):
            ml_results = train_ml_model(df, params, entry_indices)
            model = ml_results['model']
            scaler = ml_results['scaler']
            feature_df = ml_results['feature_df']
            feature_importances = ml_results['feature_importances']
            used_params.update(ml_results['used_params'])

        # --- ЭТАП СКРИНИНГА ---
        # Если включен режим скрининга, фильтруем сигналы, оставляя только "перспективные"
        if screening_mode:
            used_params.add("screening_mode")
            promising_long_mask = df['promising_long'].values
            promising_short_mask = df['promising_short'].values
        else: # Иначе создаем маски, которые пропускают все сигналы
            promising_long_mask = np.ones(len(df), dtype=bool)
            promising_short_mask = np.ones(len(df), dtype=bool)

        if not entry_indices: # Если нет сигналов, выходим
            return self.calculate_metrics()

        current_idx = 0
        signal_queue = sorted(list(set(signal_indices))) # Уникальные и отсортированные

        while signal_queue and signal_queue[0] < len(df) -1:
            # Находим первый подходящий сигнал
            try:
                next_signal_idx = next(s_idx for s_idx in signal_queue if s_idx > current_idx)
            except StopIteration:
                break # Больше нет подходящих сигналов
            
            # Удаляем все сигналы, которые мы уже прошли (включая текущий)
            signal_queue = [s for s in signal_queue if s > next_signal_idx]

            # --- Логика с подтверждением на i и входом на i+1 ---
            analysis_idx = next_signal_idx
            if analysis_idx >= len(df) - 1:
                continue # Не можем войти, так как это предпоследняя свеча

            direction = None
            entry_idx = -1
            entry_price = -1

            # --- ПРОВЕРКА СИГНАЛА МОДЕЛЬЮ (если модель есть) ---
            if model:
                ml_features_for_trade = None # type: ignore
                # Подготавливаем признаки для текущего сигнала
                current_features = feature_df.loc[[analysis_idx]]
                current_features_scaled = scaler.transform(current_features)
                
                # Предсказываем класс (1 - хороший long, 0 - плохой)
                prediction = model.predict(current_features_scaled)[0]
                if prediction != 1:
                    current_idx = analysis_idx # Сдвигаем индекс
                    self.ml_rejected_signals.append(analysis_idx)
                    continue # Сигнал отфильтрован моделью, переходим к следующему
                else:
                    # Если сигнал одобрен, сохраняем его признаки для отображения на графике
                    ml_features_for_trade = current_features.iloc[0].to_dict()

            # Логика входа теперь всегда "Вилка отложенных ордеров"
            base_price = close_prices[analysis_idx]
            long_level = base_price * (1 + bracket_offset_pct / 100)
            short_level = base_price * (1 - bracket_offset_pct / 100)

            # Ищем вход в "вилке"
            entry_idx, entry_price, direction_str = find_bracket_entry(
                start_idx=analysis_idx + 1,
                timeout=bracket_timeout_candles,
                long_level=long_level,
                short_level=short_level,
                high_prices=high_prices,
                low_prices=low_prices,
                open_prices=open_prices,
                hldir_values=hldir_values
            )
            # Направление и цена входа уже определены функцией find_bracket_entry
            direction = direction_str if direction_str != "none" else None
            # Добавляем параметры в used_params только если вход по вилке состоялся
            if direction:
                used_params.update({"bracket_offset_pct", "bracket_timeout_candles"})
            # --- Общая логика открытия и закрытия позиции ---
            if direction and entry_idx < len(df):
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, direction)
                
                # Numba-ускоренный поиск точки выхода
                exit_idx, exit_reason, exit_price = find_first_exit(
                    entry_idx=entry_idx,
                    direction_is_long=(direction == 'long'),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    use_climax_exit=use_climax_exit,
                    climax_exit_threshold=climax_exit_threshold,
                    high_prices=high_prices,
                    low_prices=low_prices,
                    close_prices=close_prices,
                    long_climax_scores=long_climax_scores,
                    short_climax_scores=short_climax_scores,
                    entry_price=entry_price
                )

                if aggressive_mode and exit_idx != -1:
                    next_signal_in_trade = next((s for s in signal_queue if entry_idx < s < exit_idx), None)
                    used_params.add("aggressive_mode")
                    if next_signal_in_trade:
                        exit_idx = next_signal_in_trade - 1 # Выходим на свече перед новым сигналом
                        exit_reason = 'new_signal'
                        exit_price = close_prices[exit_idx]

                if exit_idx != -1 and exit_idx >= entry_idx: # Убедимся, что сделка длилась хотя бы одну свечу

                    # Записываем сделку
                    trade = {
                        'entry_idx': entry_idx,
                        'entry_price': entry_price,
                        'direction': direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_idx': int(exit_idx),
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'signal_idx_for_trade': analysis_idx, # Индекс сигнальной свечи
                        'ml_features': ml_features_for_trade if model and ml_features_for_trade is not None else None # Признаки, использованные моделью
                    }
                    self.trades.append(trade)
                    current_idx = trade['exit_idx']
                else:
                    # Если входа по вилке не произошло (таймаут), или выхода по другим причинам,
                    # сдвигаем current_idx, чтобы избежать зацикливания.
                    # Для вилки, если вход был, но выхода нет, current_idx уже сдвинут в `find_first_exit`.
                    # Если входа не было, entry_idx = -1, и мы сдвигаем current_idx на analysis_idx.
                    current_idx = entry_idx if entry_idx != -1 else analysis_idx

        # --- Конец векторизованной логики ---

        # Рассчитываем PnL для всех сделок после цикла
        for trade in self.trades:
            if trade['direction'] == 'long':
                pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            else: # short
                pnl = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
            
            pnl_amount = self.position_size * pnl - (self.position_size * self.commission) * 2
            trade['pnl'] = pnl_amount
            self.pnl_history.append(pnl_amount)
            
            balance_value = self.initial_balance + pnl_amount if len(self.balance_history) == 0 else self.balance_history[-1] + pnl_amount
            self.balance_history.append(balance_value)

        return self.calculate_metrics(used_params, feature_importances, model, scaler, feature_df)

    def calculate_metrics(self, used_params=None, feature_importances=None, model=None, scaler=None, feature_df=None):
        """Вычисляет итоговые метрики после симуляции."""
        # Рассчитываем метрики с использованием векторизованных операций
        if self.pnl_history:            
            pnl_array = np.array(self.pnl_history)
            total_trades = len(pnl_array)
            winning_trades = np.sum(pnl_array > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = np.sum(pnl_array)
            
            # Рассчитываем дополнительные метрики
            avg_pnl = np.mean(pnl_array)
            profits = pnl_array[pnl_array > 0]
            losses = np.abs(pnl_array[pnl_array < 0])
            total_profit = np.sum(profits) if len(profits) > 0 else 0
            total_loss = np.sum(losses) if len(losses) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 1e-9 else float('inf')
            
            # Рассчитываем максимальную просадку
            combined_balance = [self.initial_balance] + self.balance_history
            balance_array = np.array(combined_balance)
            running_max = np.maximum.accumulate(balance_array)
            drawdowns = (running_max - balance_array) / running_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            total_trades = 0
            winning_trades = 0
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
            profit_factor = 0
            max_drawdown = 0
        
        # Обеспечиваем, что все значения в результатах являются обычными Python или NumPy значениями
        results = {
           'total_signals': int(self.total_signals),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(profit_factor),
            'trades': self.trades,
            'pnl_history': self.pnl_history,
            'balance_history': self.balance_history,
            'final_balance': self.balance_history[-1] if self.balance_history else self.initial_balance,
            'used_params': list(used_params) if used_params else [],
            'feature_importances': feature_importances,
            'ml_rejected_signals': self.ml_rejected_signals,
            # --- ИСПРАВЛЕНИЕ: Удаляем несериализуемые объекты ---
            # Эти объекты (модель, скейлер, датафрейм) не могут быть сохранены в JSON.
            # Они будут заново созданы на странице "Анализ" при необходимости.
            'ml_model': None, # Явно устанавливаем в None
            'ml_scaler': None,
            'ml_features_df': None
        }
        results['is_ml_simulation'] = bool(model)
        
        return results

@numba.jit(nopython=True, cache=True)
def find_first_exit(
    entry_idx: int,
    direction_is_long: bool,
    stop_loss: float,
    take_profit: float,
    use_climax_exit: bool,
    climax_exit_threshold: float,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    long_climax_scores: np.ndarray,
    short_climax_scores: np.ndarray,
    entry_price: float
):
    """
    Numba-ускоренная функция для поиска первой точки выхода.
    Итерирует по свечам после входа и возвращает первую сработавшую причину выхода.
    """
    for i in range(entry_idx, len(high_prices)):
        # Проверка выхода по Stop Loss / Take Profit
        if direction_is_long:
            if low_prices[i] <= stop_loss:
                return i, 'stop_loss', stop_loss
            if high_prices[i] >= take_profit:
                return i, 'take_profit', take_profit
        else: # short
            if high_prices[i] >= stop_loss:
                return i, 'stop_loss', stop_loss
            if low_prices[i] <= take_profit:
                return i, 'take_profit', take_profit

        # Проверка выхода по Climax Score
        if use_climax_exit:
            if direction_is_long:
                # Выход только если цена выше цены входа (в прибыли)
                if long_climax_scores[i] > climax_exit_threshold and close_prices[i] > entry_price:
                    return i, 'climax_exit', close_prices[i]
            else: # short
                # Выход только если цена ниже цены входа (в прибыли)
                if short_climax_scores[i] > climax_exit_threshold and close_prices[i] < entry_price:
                    return i, 'climax_exit', close_prices[i]

    return len(high_prices) - 1, 'end_of_data', close_prices[-1]

@numba.jit(nopython=True, cache=True)
def find_bracket_entry(
    start_idx: int,
    timeout: int,
    long_level: float,
    short_level: float,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    open_prices: np.ndarray,
    hldir_values: np.ndarray
):
    """
    Numba-ускоренная функция для поиска первого входа по "вилке".
    """
    end_idx = min(start_idx + timeout, len(high_prices))
    for i in range(start_idx, end_idx):
        high = high_prices[i]
        low = low_prices[i]

        hit_long = high >= long_level
        hit_short = low <= short_level

        # Случай 1: Однозначный пробой в одну сторону
        if hit_long and not hit_short:
            # Вход по long. Если open > long_level, то это проскальзывание.
            entry_price = max(long_level, open_prices[i])
            return i, entry_price, "long"
        
        if hit_short and not hit_long:
            # Вход по short. Если open < short_level, то это проскальзывание.
            entry_price = min(short_level, open_prices[i])
            return i, entry_price, "short"

        # Случай 2: Неоднозначный пробой в обе стороны за одну свечу
        if hit_long and hit_short:
            # Используем HLdir для принятия решения
            if hldir_values[i] == 1: # Если HLdir указывает на силу покупателей
                entry_price = max(long_level, open_prices[i])
                return i, entry_price, "long"
            else: # Если HLdir == 0, указывает на силу продавцов
                entry_price = min(short_level, open_prices[i])
                return i, entry_price, "short"

    return -1, -1.0, "none" # Тайм-аут или одновременный пробой

def run_grouped_trading_simulation(df, params, screening_mode=False):
    """
    Запускает симуляцию для DataFrame с несколькими инструментами,
    группируя по 'Symbol' и агрегируя результаты.
    """
    all_results = []
    for symbol, group_df in df.groupby('Symbol'):
        # Важно сбросить индекс, чтобы внутри симулятора индексация была с 0
        group_df_reset = group_df.reset_index(drop=True)
        
        # Запускаем симуляцию для одного инструмента
        result = run_trading_simulation(group_df_reset, params, screening_mode)
        
        # Корректируем индексы сделок, чтобы они соответствовали исходному `group_df`
        original_indices = group_df.index
        for trade in result['trades']:
            trade['entry_idx'] = original_indices[trade['entry_idx']]
            trade['exit_idx'] = original_indices[trade['exit_idx']]
            trade['signal_idx'] = original_indices[trade['signal_idx']]
        
        all_results.append(result)

    # Агрегируем результаты
    if not all_results:
        # Возвращаем пустую структуру, если не было результатов
        sim = TradingSimulator()
        return sim.calculate_metrics()

    # Собираем все сделки и сортируем их по времени входа
    aggregated_trades = sorted([trade for res in all_results for trade in res['trades']], key=lambda x: x['entry_idx'])
    
    # --- Расчет объединенной истории PNL ---
    # Собираем PNL из всех результатов
    final_pnl_history = []
    for res in all_results:
        final_pnl_history.extend(res['pnl_history'])

    total_trades = len(aggregated_trades)
    winning_trades = sum(1 for pnl in final_pnl_history if pnl > 0)    
    
    initial_balance = params.get("position_size", 100.0)
    combined_balance = [initial_balance]

    current_balance = initial_balance
    for pnl in final_pnl_history:
        current_balance += pnl
        combined_balance.append(current_balance)

    # --- Расчет максимальной просадки ---
    balance_array = np.array(combined_balance)
    running_max = np.maximum.accumulate(balance_array)
    drawdowns = (running_max - balance_array) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # --- Создаем и возвращаем результаты ---
    results = {
        'total_signals': sum(res['total_signals'] for res in all_results),
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_pnl': sum(final_pnl_history),

        'avg_pnl': np.mean(final_pnl_history) if final_pnl_history else 0, # type: ignore
        'max_drawdown': max_drawdown,
        'profit_factor': np.sum(np.array(final_pnl_history)[np.array(final_pnl_history) > 0]) / np.abs(np.sum(np.array(final_pnl_history)[np.array(final_pnl_history) < 0])) if any(np.array(final_pnl_history) < 0) else float('inf'),
        'trades': aggregated_trades,
        'pnl_history': final_pnl_history,
    }
    return results

def run_trading_simulation(df, params, screening_mode=False):


    """
    Запуск симуляции торговли
     
    Args:
        df: DataFrame с рыночными данными
        params: параметры стратегии
        
    Returns:
        результаты симуляции
    """
    # print("\n--- Запуск симуляции со следующими параметрами: ---")
    # pprint.pprint(params)
    # print("---------------------------------------------------\n")
 
    # Создаем копию, чтобы не изменять оригинальный словарь параметров, который может использоваться в Optuna
    params_copy = params.copy()

    # Извлекаем параметры для симулятора и округляем float параметры до 2 знаков после запятой
    # Используем pop, чтобы эти параметры не попали в generate_signals
    position_size = round(params_copy.pop("position_size", 100.0), 2)
    commission = round(params_copy.get("commission", 0.1), 3)  # Используем 3 знака для комиссии из-за малых значений
    stop_loss_pct = round(params_copy.get("stop_loss_pct", 2.0), 2)
    take_profit_pct = round(params_copy.get("take_profit_pct", 4.0), 2)
    aggressive_mode = params_copy.get("aggressive_mode", False) # Получаем режим симуляции
    
    # Новые параметры для "вилки"
    params_copy["bracket_offset_pct"] = round(params_copy.get("bracket_offset_pct", 0.5), 2)
    params_copy["bracket_timeout_candles"] = int(params_copy.get("bracket_timeout_candles", 5))
    
    # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Явное добавление всех ML-параметров ---
    # Явно добавляем все ML-параметры (как для признаков, так и для классификатора) в `params_copy` из `params`.
    # Это гарантирует, что и `generate_signals`, и `train_ml_model` внутри симулятора получат все необходимые данные,
    # особенно при запуске со страницы "Анализ" после загрузки профиля из "Аналитики".
    ml_feature_keys = [
        "classifier_type", "catboost_iterations", "catboost_depth", "catboost_learning_rate",
        "prints_analysis_period", "prints_threshold_ratio", "m_analysis_period",
        "m_threshold_ratio", "hldir_window", "hldir_offset"
    ]
    for key in ml_feature_keys:
        # `params` - это оригинальный словарь, который может содержать ML-параметры,
        # собранные функцией get_strategy_parameters.
        if key in params:
            params_copy[key] = params[key]

    # Создаем симулятор
    simulator = TradingSimulator(
        position_size=position_size,
        commission=commission,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
        # Всегда используем HLdir, если он присутствует в данных
    )
    
    # Запускаем симуляцию
    results = simulator.simulate_trades(df, params_copy, aggressive_mode, screening_mode=screening_mode)
    
    return results