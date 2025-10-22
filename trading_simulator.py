import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signal_generator import generate_signals # type: ignore

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
       """
       self.position_size = position_size
       self.commission = commission / 100  # преобразуем в доли
       self.stop_loss_pct = stop_loss_pct
       self.take_profit_pct = take_profit_pct
       self.trades = []
       self.pnl_history = []
       self.balance_history = []
       self.current_position = None
       self.initial_balance = position_size # type: ignore
        
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

    def simulate_trades(self, df, params, aggressive_mode=False):
        """
        Симуляция торговли

        Args:
            df: DataFrame с рыночными данными
            params: параметры стратегии
            aggressive_mode: если True, позволяет открывать новые сделки, не дожидаясь закрытия старых

        Returns: словарь с результатами симуляции
        """
        bracket_offset_pct = params.get("bracket_offset_pct", 0.5)
        bracket_timeout_candles = params.get("bracket_timeout_candles", 5)

        # Начинаем собирать использованные параметры
        used_params = {"position_size", "commission", "stop_loss_pct", "take_profit_pct"}


        # --- ИСПРАВЛЕНИЕ: Генерируем все индикаторы и сигналы ОДИН РАЗ ---
        # base_signal_only=False гарантирует, что мы получим и базовые сигналы,
        # и все индикаторы, которые могут понадобиться для ML в качестве признаков.
        signal_indices, _, signal_gen_used_params = generate_signals(df, params)
        self.total_signals = len(signal_indices) # Store total signals before ML filtering
        used_params.update(signal_gen_used_params)

        # --- ИСПРАВЛЕНИЕ: Генерируем DataFrame со всеми индикаторами для ML ---
        # Вызываем generate_signals с return_indicators=True, чтобы получить df с признаками
        signal_indices, df_with_features, signal_gen_used_params = generate_signals(df, params, return_indicators=True)
        if df_with_features is None: # На случай, если что-то пошло не так
            df_with_features = df.copy() # Возвращаемся к исходному df
        # --- НОВЫЙ БЛОК: Генерируем ML-признаки, если модель будет использоваться ---
        if params.get('ml_model_bundle') and params.get('use_ml_filter'):
            from ml_model_handler import generate_features # Локальный импорт
            df_with_features = generate_features(df_with_features, params)
            import torch # Локальный импорт для PyTorch
        used_params.update(signal_gen_used_params)

        open_prices = df_with_features['open'].values
        high_prices = df_with_features['high'].values
        low_prices = df_with_features['low'].values
        close_prices = df_with_features['close'].values
        hldir_values = df_with_features['HLdir'].values if 'HLdir' in df_with_features.columns else np.zeros_like(open_prices)
        xp = np # Используется для np.ones

        # 2. Итерируемся по сигналам, а не по всем свечам

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

            # --- НОВЫЙ БЛОК: Фильтрация сигналов с помощью ML-модели ---
            ml_bundle = params.get('ml_model_bundle')
            prediction = [1] # По умолчанию считаем сигнал хорошим
            if ml_bundle and params.get('use_ml_filter'):
                model = ml_bundle['model']
                scaler = ml_bundle['scaler']
                feature_names = ml_bundle['feature_names']
                numerical_features = ml_bundle['numerical_features']
                categorical_features = ml_bundle.get('categorical_features', [])

                # --- НОВЫЙ БЛОК: Универсальная логика предсказания ---
                model_type = ml_bundle.get('model_type', 'CatBoost')
                signal_features_df = df_with_features.loc[[analysis_idx]]

                if model_type == 'CatBoost':
                    features_for_prediction = signal_features_df.copy()
                    num_features_in_df = [f for f in numerical_features if f in features_for_prediction.columns]
                    if scaler and num_features_in_df:
                        features_for_prediction[num_features_in_df] = scaler.transform(features_for_prediction[num_features_in_df])
                    for col in categorical_features:
                        if col in features_for_prediction.columns:
                            features_for_prediction[col] = features_for_prediction[col].astype(int)
                    prediction = model.predict(features_for_prediction[feature_names])

                elif model_type == 'NeuralNetwork':
                    features_for_prediction = signal_features_df.copy()
                    num_features_in_df = [f for f in numerical_features if f in features_for_prediction.columns]
                    if scaler and num_features_in_df:
                        features_for_prediction[num_features_in_df] = scaler.transform(features_for_prediction[num_features_in_df])
                    
                    # One-hot encoding для категориальных признаков
                    features_nn = pd.get_dummies(features_for_prediction, columns=categorical_features, drop_first=True)
                    # Приводим столбцы к тому же порядку, что и при обучении
                    nn_columns = ml_bundle.get('nn_columns')
                    if nn_columns:
                        features_nn = features_nn.reindex(columns=nn_columns, fill_value=0)

                    # Предсказание с помощью PyTorch
                    model.eval() # Переводим модель в режим оценки
                    with torch.no_grad():
                        X_tensor = torch.tensor(features_nn.values, dtype=torch.float32)
                        output = model(X_tensor)
                        # Применяем сигмоиду и порог 0.5 для получения бинарного предсказания
                        pred_prob = torch.sigmoid(output)
                        prediction = (pred_prob > 0.5).int().numpy()

                if prediction[0] == 0: # Если модель предсказывает провал (0)
                    # --- НОВЫЙ БЛОК: Симулируем пропущенную сделку для анализа ---
                    # Это нужно, чтобы потом можно было посмотреть, от каких сделок модель отказалась.
                    
                    # 1. Ищем потенциальный вход (та же логика, что и для реальных сделок)
                    temp_base_price = open_prices[analysis_idx + 1]
                    temp_long_level = temp_base_price * (1 + bracket_offset_pct / 100)
                    temp_short_level = temp_base_price * (1 - bracket_offset_pct / 100)
                    temp_use_hldir = params.get("use_hldir_on_conflict", True)
                    temp_simulate_slippage = params.get("simulate_slippage", True)

                    temp_entry_idx, temp_entry_price, temp_direction_str = find_bracket_entry(
                        start_idx=analysis_idx + 1, timeout=bracket_timeout_candles,
                        long_level=temp_long_level, short_level=temp_short_level,
                        use_hldir_on_conflict=temp_use_hldir, simulate_slippage=temp_simulate_slippage,
                        hldir_values=hldir_values, high_prices=high_prices,
                        low_prices=low_prices, open_prices=open_prices
                    )

                    if temp_direction_str != "none":
                        # 2. Если вход был бы, ищем потенциальный выход
                        temp_sl, temp_tp = self.calculate_stop_loss_take_profit(temp_entry_price, temp_direction_str)
                        temp_exit_idx, temp_exit_reason, temp_exit_price = find_first_exit(
                            entry_idx=temp_entry_idx, direction_is_long=(temp_direction_str == 'long'),
                            stop_loss=temp_sl, take_profit=temp_tp,
                            high_prices=high_prices, low_prices=low_prices, close_prices=close_prices
                        )

                        # 3. Сохраняем "виртуальную" сделку
                        skipped_trade = {
                            'entry_idx': temp_entry_idx, 'entry_price': temp_entry_price,
                            'direction': temp_direction_str, 'stop_loss': temp_sl, 'take_profit': temp_tp,
                            'exit_idx': int(temp_exit_idx), 'exit_price': temp_exit_price,
                            'exit_reason': temp_exit_reason, 'signal_idx_for_trade': analysis_idx,
                            'pnl': 0, # PnL не считаем, т.к. сделки не было
                            'skipped_by_ml': True # Флаг, что сделка пропущена
                        }
                        self.trades.append(skipped_trade) # Добавляем в основной список сделок

                    # --- ИСПРАВЛЕНИЕ: Обновляем current_idx даже для пропущенных сделок ---
                    # Это предотвращает наложение сделок, так как следующий сигнал будет искаться только после завершения этой "виртуальной" сделки.
                    current_idx = temp_exit_idx if temp_direction_str != "none" else analysis_idx
                    continue
            # --- Конец блока ML-фильтрации ---

            direction = None
            entry_idx = -1
            entry_price = -1

            # --- ИСПРАВЛЕНИЕ: Базовая цена для "вилки" - это цена открытия СЛЕДУЮЩЕЙ свечи, а не закрытия сигнальной ---
            # Это более реалистично, так как сигнал появляется после закрытия свечи `analysis_idx`, а ордера выставляются относительно открытия следующей.
            base_price = open_prices[analysis_idx + 1] # Используем open следующей свечи
            long_level = base_price * (1 + bracket_offset_pct / 100) # type: ignore
            short_level = base_price * (1 - bracket_offset_pct / 100) # type: ignore

            use_hldir_on_conflict_flag = params.get("use_hldir_on_conflict", True)
            simulate_slippage_flag = params.get("simulate_slippage", True)
            # Ищем вход в "вилке"
            entry_idx, entry_price, direction_str = find_bracket_entry(
                start_idx=analysis_idx + 1,
                timeout=bracket_timeout_candles,
                long_level=long_level,
                short_level=short_level,
                use_hldir_on_conflict=use_hldir_on_conflict_flag,
                simulate_slippage=simulate_slippage_flag,
                hldir_values=hldir_values,
                high_prices=high_prices,
                low_prices=low_prices, # type: ignore
                open_prices=open_prices
            )
            # Направление и цена входа уже определены функцией find_bracket_entry
            direction = direction_str if direction_str != "none" else None
            # Добавляем параметры в used_params только если вход по вилке состоялся
            if direction:
                if use_hldir_on_conflict_flag:
                    used_params.add("use_hldir_on_conflict")
                if simulate_slippage_flag:
                    used_params.add("simulate_slippage")
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
                    high_prices=high_prices,
                    low_prices=low_prices,
                    close_prices=close_prices
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
                        'ml_features': None
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
        # --- ИЗМЕНЕНИЕ: Отделяем реальные сделки от пропущенных для расчета PnL ---
        real_trades = [t for t in self.trades if not t.get('skipped_by_ml')]
        skipped_trades = [t for t in self.trades if t.get('skipped_by_ml')]

        # PnL и баланс считаем только для реальных сделок
        for trade in real_trades: # --- ИСПРАВЛЕНИЕ: Итерируемся только по реальным сделкам ---
            if trade['direction'] == 'long':
                pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            else: # short
                pnl = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
            
            pnl_amount = self.position_size * pnl - (self.position_size * self.commission) * 2
            trade['pnl'] = pnl_amount
            self.pnl_history.append(pnl_amount)
            balance_value = self.initial_balance + pnl_amount if not self.balance_history else self.balance_history[-1] + pnl_amount
            self.balance_history.append(balance_value)

        return self.calculate_metrics(used_params)

    def calculate_metrics(self, used_params=None):
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
            pnl_array = np.array([]) # <--- ИСПРАВЛЕНИЕ: Инициализируем pnl_array как пустой массив
            # Если нет сделок, все метрики равны 0
            total_trades = 0
            winning_trades = 0
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
            profit_factor = 0
            max_drawdown = 0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            expectancy = 0.0
            avg_pnl_per_trade = 0.0
        
        # Calculate Sharpe Ratio
        # Assuming risk-free rate is 0 and 252 trading days for annualization
        if len(pnl_array) >= 2 and pnl_array.std() != 0:
            sharpe_ratio = pnl_array.mean() / pnl_array.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Calculate Sortino Ratio
        negative_pnl = pnl_array[pnl_array < 0]
        if len(negative_pnl) >= 2 and negative_pnl.std() != 0:
            sortino_ratio = pnl_array.mean() / negative_pnl.std() * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # Calculate Expectancy
        if total_trades > 0:
            avg_win = profits.mean() if len(profits) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            loss_rate = (total_trades - winning_trades) / total_trades
            expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
            avg_pnl_per_trade = total_pnl / total_trades
        else:
            profits = np.array([]) # Инициализируем, чтобы избежать ошибок ниже
            total_trades = 0
            winning_trades = 0
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
            profit_factor = 0
            max_drawdown = 0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            expectancy = 0.0
            avg_pnl_per_trade = 0.0
        
        # Обеспечиваем, что все значения в результатах являются обычными Python или NumPy значениями
        results = {
           'total_signals': self.total_signals if hasattr(self, 'total_signals') else 0,
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(profit_factor),
            'sharpe_ratio': float(sharpe_ratio), # NEW
            'sortino_ratio': float(sortino_ratio), # NEW
            'expectancy': float(expectancy), # NEW
            'avg_pnl_per_trade': float(avg_pnl_per_trade), # NEW
            'trades': self.trades,
            'pnl_history': self.pnl_history,
            'balance_history': self.balance_history,
            'final_balance': self.balance_history[-1] if self.balance_history else self.initial_balance,
            'used_params': list(used_params) if used_params else [],            
            # --- ИСПРАВЛЕНИЕ: Удаляем несериализуемые объекты ---
            # Эти объекты (модель, скейлер, датафрейм) не могут быть сохранены в JSON.
            # Они будут заново созданы на странице "Анализ" при необходимости.
            'ml_model': None, # Явно устанавливаем в None
            'ml_scaler': None,
            'ml_features_df': None
        }        
        results['is_ml_simulation'] = False
        
        return results

@numba.jit(nopython=True, cache=True)
def find_first_exit(
    entry_idx: int,
    direction_is_long: bool,
    stop_loss: float,
    take_profit: float,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray
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

    return len(high_prices) - 1, 'end_of_data', close_prices[-1]

@numba.jit(nopython=True, cache=True)
def find_bracket_entry(
    start_idx: int,
    timeout: int,
    long_level: float,
    short_level: float,
    simulate_slippage: bool,
    use_hldir_on_conflict: bool,
    hldir_values: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    open_prices: np.ndarray
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
            if simulate_slippage:
                entry_price = max(long_level, open_prices[i])
            else:
                entry_price = long_level
            return i, entry_price, "long"
        
        if hit_short and not hit_long:
            # Вход по short. Если open < short_level, то это проскальзывание.
            if simulate_slippage:
                entry_price = min(short_level, open_prices[i])
            else:
                entry_price = short_level
            return i, entry_price, "short"

        # Случай 2: Неоднозначный пробой в обе стороны за одну свечу
        if hit_long and hit_short:
            # Если опция использования HLdir включена
            if use_hldir_on_conflict:
                if hldir_values[i] == 1:  # HLdir = 1 указывает на лонг
                    if simulate_slippage:
                        entry_price = max(long_level, open_prices[i])
                    else:
                        entry_price = long_level
                    return i, entry_price, "long"
                elif hldir_values[i] == 0:  # HLdir = 0 указывает на шорт
                    if simulate_slippage:
                        entry_price = min(short_level, open_prices[i])
                    else:
                        entry_price = short_level
                    return i, entry_price, "short"
            return -1, -1.0, "none"

    return -1, -1.0, "none" # Тайм-аут

def run_grouped_trading_simulation(df, params):
    """
    Запускает симуляцию для DataFrame с несколькими инструментами,
    группируя по 'Symbol' и агрегируя результаты.
    """
    all_results = []
    for symbol, group_df in df.groupby('Symbol'):
        # Важно сбросить индекс, чтобы внутри симулятора индексация была с 0
        group_df_reset = group_df.reset_index(drop=True)
        
        # Запускаем симуляцию для одного инструмента
        result = run_trading_simulation(group_df_reset, params)
        
        # Корректируем индексы сделок, чтобы они соответствовали исходному `group_df`
        original_indices = group_df.index
        for trade in result['trades']:
            trade['entry_idx'] = original_indices[trade['entry_idx']]
            trade['exit_idx'] = original_indices[trade['exit_idx']]
            trade['signal_idx_for_trade'] = original_indices[trade['signal_idx_for_trade']]
        
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

def run_trading_simulation(df, params):


    """
    Запуск симуляции торговли
     
    Args:
        df: DataFrame с рыночными данными
        params: параметры стратегии
        
    Returns:
        результаты симуляции
    """
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

    # --- ИСПРАВЛЕНИЕ: Явно передаем флаг использования ML-фильтра в копию параметров ---
    # Это гарантирует, что симулятор "увидит" этот флаг.
    params_copy["use_ml_filter"] = params.get("use_ml_filter", False)
    
    # Создаем симулятор
    simulator = TradingSimulator(
        position_size=position_size,
        commission=commission,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    # Запускаем симуляцию
    results = simulator.simulate_trades(df, params_copy, aggressive_mode)
    
    return results