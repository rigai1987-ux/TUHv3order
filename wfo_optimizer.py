"""
Модуль для проведения пошаговой оптимизации (Walk-Forward Optimization).

Этот модуль использует `optuna_optimizer` для поиска лучших параметров на обучающих
периодах (in-sample) и затем тестирует их на следующих, невидимых периодах
(out-of-sample) для оценки робастности и адаптивности стратегии.
"""

import os
import pandas as pd
from datetime import timedelta
from joblib import Parallel, delayed
import optuna_optimizer
from trading_simulator import run_trading_simulation
import streamlit as st
import optuna
import numpy as np
# Импортируем ML-функции
from signal_generator import generate_signals
from ml_model_handler import label_all_signals, generate_features, train_ml_model

from strategy_objectives import trading_strategy_objective_sqn, trading_strategy_objective_hft_score, trading_strategy_objective_ml_data_quality, trading_strategy_objective_ml, trading_strategy_multi_objective_ml, trading_strategy_multi_objective
# --- ИСПРАВЛЕНИЕ: Импортируем ВСЕ возможные целевые функции ---
# Это необходимо, чтобы функция _wfo_optimization_task могла восстановить любую
# целевую функцию по её имени (strategy_func_name) в параллельных процессах.
# Без этого globals().get(strategy_func_name) возвращал None, что приводило к ошибке.
from strategy_objectives import (
    trading_strategy_objective_sqn, trading_strategy_objective_hft_score,
    trading_strategy_objective_ml_data_quality, trading_strategy_objective_ml,
    trading_strategy_multi_objective_ml, trading_strategy_multi_objective,
    trading_strategy_objective_equity_curve_linearity
)
def _wfo_optimization_task(task_params):
    """
    Вспомогательная функция для выполнения одной задачи оптимизации в параллельном режиме.
    Принимает и возвращает словарь, чтобы быть совместимой с joblib.
    """
    step_count = task_params['step_count']
    total_steps = task_params['total_steps']
    opt_params = task_params['opt_params']
    opt_params['n_jobs'] = 1 # ВАЖНО: Каждая задача WFO выполняется в одном потоке

    # --- ИСПРАВЛЕНИЕ: Восстанавливаем объект функции по имени, не удаляя имя из словаря. ---
    # Это решает проблему с сериализацией (pickling) и гарантирует, что 'strategy_func_name'
    # останется в `task['opt_params']` для использования на этапе сборки результатов.
    strategy_func_name = opt_params['strategy_func_name']
    opt_params['strategy_func'] = globals().get(strategy_func_name)
    
    print(f"Starting parallel WFO optimization for step {step_count}/{total_steps}...")
    in_sample_opt_results = optuna_optimizer.run_optimization(opt_params)
    print(f"Finished parallel WFO optimization for step {step_count}/{total_steps}.")
    
    task_params['in_sample_opt_results'] = in_sample_opt_results
    return task_params


def run_wfo_parallel(
    data: pd.DataFrame,
    wfo_params: dict,
    opt_params: dict,
):
    """
    Запускает WFO, выполняя фазу оптимизации для всех шагов параллельно.
    Это значительно быстрее, но не предоставляет пошагового UI.
    """
    # 1. --- Фаза подготовки: нарезка данных и создание задач ---
    st.info("Фаза 1: Подготовка данных и создание задач для WFO...")
    
    train_period = wfo_params['train_period']
    test_period = wfo_params['test_period']
    step_period = wfo_params['step_period']
    wfo_unit = wfo_params.get('wfo_unit', 'Дни')

    start_date_data = data['datetime'].min()
    end_date_data = data['datetime'].max()

    if wfo_unit == 'Дни':
        total_duration = (end_date_data - start_date_data).days
        train_duration = timedelta(days=train_period)
        test_duration = timedelta(days=test_period)
        step_duration = timedelta(days=step_period)
    else: # Часы
        total_duration = int((end_date_data - start_date_data).total_seconds() / 3600)
        train_duration = timedelta(hours=train_period)
        test_duration = timedelta(hours=test_period)
        step_duration = timedelta(hours=step_period)

    if total_duration < (train_period + test_period):
        st.error("Недостаточно данных для проведения WFO с заданными размерами окон.")
        return {"summary": [], "aggregated_metrics": {}, "equity_curve": pd.DataFrame()}

    total_walk_duration = total_duration - train_period - test_period
    total_steps = (total_walk_duration // step_period) + 1 if total_walk_duration >= 0 else 0

    tasks = []
    current_start_date = start_date_data
    for i in range(total_steps):
        train_start_date = current_start_date
        train_end_date = train_start_date + train_duration
        test_start_date = train_end_date
        test_end_date = test_start_date + test_duration

        train_data = data[(data['datetime'] >= train_start_date) & (data['datetime'] < train_end_date)].copy().reset_index(drop=True)
        test_data = data[(data['datetime'] >= test_start_date) & (data['datetime'] < test_end_date)].copy().reset_index(drop=True)

        if train_data.empty or test_data.empty:
            current_start_date += step_duration
            continue

        # Подготовка параметров для конкретной задачи
        task_opt_params = opt_params.copy()
        task_opt_params['data'] = train_data
        task_opt_params['n_trials'] = wfo_params.get('trials_per_step', 25)
        if 'base_settings' not in task_opt_params: task_opt_params['base_settings'] = {}
        task_opt_params['base_settings']['min_trades_threshold'] = wfo_params.get('min_trades_threshold', 10)
        # --- ИСПРАВЛЕНИЕ: Передаем имя функции, а не сам объект ---
        # Это решает проблему с сериализацией (pickling) для joblib.
        task_opt_params['strategy_func_name'] = task_opt_params.pop('strategy_func').__name__


        tasks.append({
            "step_count": i + 1,
            "total_steps": total_steps,
            "train_data": train_data,
            "test_data": test_data,
            "opt_params": task_opt_params,
            "wfo_params": wfo_params,
            "train_period_str": f"{train_start_date:%Y-%m-%d} : {train_end_date:%Y-%m-%d}",
            "test_period_str": f"{test_start_date:%Y-%m-%d} : {test_end_date:%Y-%m-%d}",
        })
        current_start_date += step_duration

    # 2. --- Фаза параллельной оптимизации ---
    n_jobs = os.cpu_count() or 1
    with st.spinner(f"Фаза 2: Запуск параллельной оптимизации на {n_jobs} ядрах. Это может занять много времени..."):
        # Используем 'loky' бэкенд. Он использует процессы вместо потоков, что обеспечивает
        # лучшую изоляцию и предотвращает конфликты между Numba-ускоренными задачами.
        # Это позволяет достичь 100% загрузки ЦП на вычислительно-интенсивных задачах.
        # ВАЖНО: Мы не передаем n_jobs в run_optimization напрямую.
        # Вместо этого, мы говорим joblib запустить `n_jobs` задач,
        # а каждая задача внутри себя будет использовать только 1 ядро.
        # Это предотвращает вложенный параллелизм и позволяет joblib
        # эффективно управлять ресурсами.
        completed_tasks = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_wfo_optimization_task)(task) for task in tasks)

    st.info("Фаза 3: Сборка результатов и тестирование на Out-of-Sample данных...")

    # 3. --- Фаза последовательного тестирования и агрегации ---
    wfo_summary = []
    all_oos_trades = []
    out_of_sample_results = []
    all_opt_errors = [] # Список для сбора ошибок из всех шагов

    for task in sorted(completed_tasks, key=lambda x: x['step_count']):
        in_sample_opt_results = task['in_sample_opt_results']
        
        if not in_sample_opt_results or not in_sample_opt_results.get('best_params'):
            st.warning(f"На шаге {task['step_count']} не найдено оптимальных параметров. Пропуск.")
            continue

        # Собираем ошибки, если они были
        if in_sample_opt_results.get('errors'):
            for error in in_sample_opt_results['errors']:
                error['wfo_step'] = task['step_count'] # Добавляем номер шага к ошибке
            all_opt_errors.extend(in_sample_opt_results['errors'])

        best_params = in_sample_opt_results['best_params']

        simulation_params = {**task['opt_params'].get('base_settings', {}), **best_params}
        ml_applied_on_step = False # Флаг для логирования
        # --- УЛУЧШЕНИЕ: Проверяем только явный флаг is_ml_wfo ---
        is_ml_objective = task['opt_params'].get('is_ml_wfo', False)
        
        # --- НОВЫЙ БЛОК: Финальное обучение ML-модели (аналогично run_wfo) ---
        if is_ml_objective:
            # Этот блок теперь будет выполняться и для обычных целевых функций, если установлен флаг is_ml_wfo
            with st.spinner(f"Шаг {task['step_count']}: Финальное обучение ML-модели на In-Sample..."):
                # 1. Генерируем сигналы и признаки на всем In-Sample отрезке с лучшими параметрами
                signal_indices_final, df_with_indicators_final, _ = generate_signals(task['train_data'], best_params, return_indicators=True)
                df_with_features_final = generate_features(df_with_indicators_final, best_params)

                # 2. Размечаем сигналы
                X_final, y_final = label_all_signals(df_with_features_final, signal_indices_final, best_params)

                # --- ИЗМЕНЕНИЕ: Проверяем, что в y_final есть и 0, и 1 для обучения ---
                if not X_final.empty and y_final.nunique() > 1 and y_final.sum() > 3:
                    # 3. Обучаем финальную модель
                    final_model_bundle = train_ml_model(X_final, y_final, best_params)
                    # 4. Добавляем модель и флаг в параметры для Out-of-Sample симуляции
                    simulation_params['ml_model_bundle'] = final_model_bundle
                    simulation_params['use_ml_filter'] = True
                    ml_applied_on_step = True
                    st.info(f"✓ Шаг {task['step_count']}: ML-модель обучена и будет применена на тесте.")
                else:
                    # --- ИЗМЕНЕНИЕ: Улучшаем логирование причины пропуска обучения ---
                    if not X_final.empty and y_final.nunique() <= 1:
                         st.warning(f"∅ Шаг {task['step_count']}: Обучение ML-модели пропущено, так как все сигналы на In-Sample отрезке имеют одинаковый исход. Тест будет без ML-фильтра.")

                    reason = ""
                    if X_final.empty:
                        reason = f"Не найдено сигналов, приведших к сделкам (найдено базовых сигналов: {len(signal_indices_final)})."
                    else:
                        reason = f"Найдено всего {y_final.sum()} успешных сигналов (требуется > 3)."
                    st.warning(f"∅ Шаг {task['step_count']}: Недостаточно данных для финального обучения ML-модели. Тест будет без ML-фильтра. Причина: {reason}")
        # --- КОНЕЦ НОВОГО БЛОКА ---

        out_of_sample_run = run_trading_simulation(task['test_data'], simulation_params)
        out_of_sample_results.append(out_of_sample_run)

        if out_of_sample_run['trades']:
            trades_df = pd.DataFrame(out_of_sample_run['trades'])
            trades_df['exit_time'] = task['test_data']['datetime'].iloc[trades_df['exit_idx']].values
            all_oos_trades.append(trades_df)

        summary_step = {
            "step": task['step_count'],
            "train_period": task['train_period_str'],
            "test_period": task['test_period_str'],
            "in_sample_metric": in_sample_opt_results['best_value'],
            "out_sample_pnl": out_of_sample_run['total_pnl'],
            "out_sample_trades": out_of_sample_run['total_trades'],
            "out_sample_win_rate": out_of_sample_run['win_rate'],
            "out_sample_max_drawdown": out_of_sample_run['max_drawdown'],
            "out_sample_profit_factor": out_of_sample_run['profit_factor'],
            "out_sample_sharpe_ratio": out_of_sample_run['sharpe_ratio'],
            "out_sample_sortino_ratio": out_of_sample_run['sortino_ratio'],
            "ml_applied": ml_applied_on_step, # Добавляем новый столбец
        }
        summary_step.update(best_params)
        wfo_summary.append(summary_step)

    st.success("Walk-Forward оптимизация (параллельный режим) завершена!")

    # Отображаем собранные ошибки после завершения всех процессов
    if all_opt_errors:
        st.error(f"Во время WFO произошло {len(all_opt_errors)} ошибок в пробах оптимизации:")
        for error_info in all_opt_errors[:10]: # Показываем первые 10
            with st.expander(f"Ошибка на шаге WFO #{error_info['wfo_step']} (проба #{error_info['trial_number']}): {error_info['error']}"):
                st.json(error_info['params'])


    # --- Агрегация финальных результатов (аналогично run_wfo) ---
    if not out_of_sample_results:
        return {"summary": [], "aggregated_metrics": {}, "equity_curve": pd.DataFrame()}

    equity_curve_df = pd.DataFrame()
    if all_oos_trades:
        full_trades_df = pd.concat(all_oos_trades).sort_values('exit_time').reset_index(drop=True)
        full_trades_df['cumulative_pnl'] = full_trades_df['pnl'].cumsum()
        equity_curve_df = full_trades_df[['exit_time', 'cumulative_pnl']]

    total_pnl = equity_curve_df['cumulative_pnl'].iloc[-1] if not equity_curve_df.empty else 0
    all_trades_count = sum(res['total_trades'] for res in out_of_sample_results)
    all_wins = sum(res['winning_trades'] for res in out_of_sample_results)
    all_pnl_history = [pnl for res in out_of_sample_results for pnl in res['pnl_history']]
    
    # Пересчитываем агрегированные метрики на основе объединенной истории PnL
    pnl_array_overall = np.array(all_pnl_history)
    
    overall_profit_factor = np.sum(pnl_array_overall[pnl_array_overall > 0]) / np.abs(np.sum(pnl_array_overall[pnl_array_overall < 0])) if np.sum(pnl_array_overall[pnl_array_overall < 0]) != 0 else float('inf')

    # Расчет максимальной просадки на основе объединенной кривой доходности
    if not equity_curve_df.empty:
        balance_array = np.array([0] + equity_curve_df['cumulative_pnl'].tolist())
        running_max = np.maximum.accumulate(balance_array)
        drawdowns = (running_max - balance_array) / running_max
        overall_max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    else:
        overall_max_drawdown = 0

    if len(pnl_array_overall) >= 2 and pnl_array_overall.std() != 0:
        overall_sharpe_ratio = pnl_array_overall.mean() / pnl_array_overall.std() * np.sqrt(252) # Annualized
    else:
        overall_sharpe_ratio = 0.0

    negative_pnl_overall = pnl_array_overall[pnl_array_overall < 0]
    if len(negative_pnl_overall) >= 2 and negative_pnl_overall.std() != 0:
        overall_sortino_ratio = pnl_array_overall.mean() / negative_pnl_overall.std() * np.sqrt(252) # Annualized
    else:
        overall_sortino_ratio = 0.0

    overall_avg_pnl_per_trade = total_pnl / all_trades_count if all_trades_count > 0 else 0.0

    # Для Expectancy
    all_profits = np.sum(pnl_array_overall[pnl_array_overall > 0])
    all_losses = np.abs(np.sum(pnl_array_overall[pnl_array_overall < 0]))
    
    # --- ИСПРАВЛЕНИЕ: Более надежный расчет Expectancy ---
    if all_trades_count > 0:
        win_rate = all_wins / all_trades_count
        loss_rate = 1 - win_rate
        
        # Средняя прибыль (только если есть прибыльные сделки)
        avg_win = all_profits / all_wins if all_wins > 0 else 0.0
        # Средний убыток (только если есть убыточные сделки)
        avg_loss = all_losses / (all_trades_count - all_wins) if (all_trades_count - all_wins) > 0 else 0.0
        
        overall_expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
    else:
        overall_expectancy = 0.0
    
    aggregated_metrics = {
        "total_out_of_sample_pnl": total_pnl,
        "total_out_of_sample_trades": all_trades_count,
        "overall_win_rate": all_wins / all_trades_count if all_trades_count > 0 else 0,
        "overall_profit_factor": overall_profit_factor,
        "overall_max_drawdown": overall_max_drawdown,
        "overall_sharpe_ratio": overall_sharpe_ratio,
        "overall_sortino_ratio": overall_sortino_ratio,
        "overall_expectancy": overall_expectancy,
        "overall_avg_pnl_per_trade": overall_avg_pnl_per_trade,
    }

    return {"summary": wfo_summary, "aggregated_metrics": aggregated_metrics, "equity_curve": equity_curve_df}

def run_wfo_with_auto_ranges(
    full_data: pd.DataFrame,
    wfo_params: dict,
    base_settings: dict,
    initial_param_space: dict,
    strategy_objective_func,
    pre_opt_period_pct: int,
    top_trials_pct_for_ranges: int,
    n_trials_pre_opt: int
):
    """
    Запускает WFO с автоматическим подбором робастных диапазонов.

    Args:
        full_data: Полный DataFrame с данными.
        wfo_params: Параметры для WFO (размеры окон, шаг и т.д.).
        base_settings: Базовые настройки симуляции (размер позиции, комиссия).
        initial_param_space: Исходные (широкие) диапазоны для оптимизации.
        strategy_objective_func: Целевая функция для оптимизации.
        pre_opt_period_pct: Процент данных для этапа "разведки".
        top_trials_pct_for_ranges: Процент лучших проб для анализа.
        n_trials_pre_opt: Количество проб для этапа "разведки".
    """
    st.header("Этап 1: Поиск робастных диапазонов (Разведка)")

    # 1. Выделяем данные для предварительной оптимизации
    cutoff_index = int(len(full_data) * (pre_opt_period_pct / 100))
    pre_opt_data = full_data.iloc[:cutoff_index].copy()
    st.info(f"Для разведки используется {pre_opt_period_pct}% данных (до {pre_opt_data['datetime'].max().strftime('%Y-%m-%d')}).")

    # 2. Запускаем широкую оптимизацию
    with st.spinner(f"Запуск широкого поиска на {n_trials_pre_opt} проб..."):
        pre_opt_params = {
            'data': pre_opt_data,
            'param_space': initial_param_space,
            'n_trials': n_trials_pre_opt,
            'strategy_func': strategy_objective_func,
            'base_settings': base_settings,
            'direction': 'maximize' # Предполагаем максимизацию для простоты
        }
        pre_opt_results = optuna_optimizer.run_optimization(pre_opt_params)

    if not pre_opt_results or not pre_opt_results.get('study'):
        st.error("Этап разведки не дал результатов. Невозможно определить робастные диапазоны.")
        return

    study = pre_opt_results['study']
    # Отбираем только завершенные и успешные пробы
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    
    if not completed_trials:
        st.error("На этапе разведки все пробы завершились с ошибкой. Невозможно определить робастные диапазоны.")
        return

    # 3. Анализируем результаты и создаем новые диапазоны
    st.subheader("Результаты этапа разведки")
    
    # Сортируем пробы по значению (целевой метрике)
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    # Определяем количество лучших проб для анализа
    num_top_trials = int(len(completed_trials) * (top_trials_pct_for_ranges / 100))
    top_trials = completed_trials[:num_top_trials]

    st.info(f"Анализ {len(top_trials)} лучших проб (топ {top_trials_pct_for_ranges}%) для определения новых диапазонов.")

    new_param_space = {}
    params_summary = []

    # --- НОВЫЙ БЛОК: Определяем абсолютные минимальные значения для параметров ---
    # Эти значения будут служить "последним рубежом", ниже которого
    # автоматический подбор диапазонов опуститься не сможет.
    ABSOLUTE_MINIMUMS = {
        "vol_period": 1, "vol_pctl": 1,
        "range_period": 1, "rng_pctl": 1,
        "natr_period": 1, "natr_min": 0.01,
        "lookback_period": 1, "min_growth_pct": -100.0,
        "stop_loss_pct": 0.01, "take_profit_pct": 0.01,
        "bracket_offset_pct": 0.01, "bracket_timeout_candles": 1,
        "ml_iterations": 10, "ml_depth": 2,
        "ml_epochs": 5, "ml_hidden_size": 16, "ml_num_hidden_layers": 1,
        "ml_batch_size": 16, "ml_dropout_rate": 0.01,
        "ml_learning_rate": 0.0001,
        "ml_prints_window": 1, "ml_labeling_timeout_candles": 1,
    }
    # --- КОНЕЦ НОВОГО БЛОКА ---

    for param_name, p_space in initial_param_space.items():
        p_type, initial_low, initial_high = p_space[0], p_space[1], p_space[2]
        if p_type in ["int", "float"] and param_name in study.best_trial.params:
            best_value = study.best_trial.params[param_name]
            initial_spread = initial_high - initial_low

            # --- НОВАЯ ЛОГИКА: Расчет "разброса" для нового диапазона ---
            # 1. Определяем, является ли исходный диапазон "узким"
            # Считаем диапазон узким, если его ширина меньше 20% от его среднего значения.
            # Добавляем 1e-6, чтобы избежать деления на ноль.
            is_narrow_range = initial_spread < ((initial_low + initial_high) / 2) * 0.2

            # 2. Рассчитываем "разброс" (spread)
            if is_narrow_range:
                # Если диапазон узкий, берем % от самого значения, чтобы его расширить
                spread = max(1 if p_type == "int" else 0.01, abs(best_value) * 0.20) # 20% от значения
            else:
                # Если диапазон широкий, берем % от его ширины, чтобы сузить
                spread = initial_spread * 0.25 # 25% от ширины

            # 3. Определяем новые границы, центрированные вокруг лучшего значения
            new_low_raw = best_value - spread
            new_high_raw = best_value + spread

            # 4. Ограничиваем новые границы
            # Получаем абсолютный минимум для этого параметра
            abs_min = ABSOLUTE_MINIMUMS.get(param_name, initial_low)

            if is_narrow_range:
                # Для узких диапазонов позволяем расширение, но не ниже абсолютного минимума.
                new_low = max(abs_min, new_low_raw)
                new_high = new_high_raw
            else:
                # Для широких диапазонов остаемся внутри исходных рамок (которые уже учитывают abs_min).
                new_low = max(initial_low, new_low_raw)
                new_high = min(initial_high, new_high_raw)

            # 5. Финальная корректировка для int и float
            if p_type == "int":
                new_low = int(np.floor(new_low))
                new_high = int(np.ceil(new_high))
                # Гарантируем, что диапазон не пустой и корректен
                if new_low >= new_high:
                    new_high = new_low + 1
                new_param_space[param_name] = (p_type, new_low, new_high)
            else: # float
                if np.isclose(new_low, new_high):
                    new_high = new_high + 0.01
                new_param_space[param_name] = (p_type, new_low, new_high)
            
            params_summary.append({"Параметр": param_name, "Старый диапазон": f"{initial_low} - {initial_high}", "Новый диапазон": f"{new_low} - {new_high}"})
        else: # Категориальные или отсутствующие в лучших пробах параметры оставляем без изменений
            new_param_space[param_name] = p_space
            if param_name not in study.best_trial.params:
                continue

    st.dataframe(pd.DataFrame(params_summary), use_container_width=True)

    # 4. Запускаем WFO с новыми, "робастными" диапазонами
    st.header("Этап 2: Запуск Walk-Forward Optimization с робастными диапазонами")
    opt_params_for_wfo = {
        'param_space': new_param_space,
        'strategy_func': strategy_objective_func,
        'direction': 'maximize', # Упрощаем для примера
        'base_settings': base_settings
    }

    # --- ИСПРАВЛЕНИЕ: Проверяем, является ли цель ML-ориентированной, и устанавливаем флаг ---
    # Это гарантирует, что на втором этапе (WFO) будет правильно обучаться и применяться ML-модель.
    is_ml_wfo = "ml" in strategy_objective_func.__name__
    if is_ml_wfo:
        opt_params_for_wfo['is_ml_wfo'] = True
        st.info("Обнаружена ML-цель. На каждом шаге WFO будет обучаться и применяться ML-фильтр.")
    
    # Запускаем стандартный параллельный WFO
    wfo_results = run_wfo_parallel(full_data, wfo_params, opt_params_for_wfo)

    # --- НОВЫЙ БЛОК: Отображение результатов WFO после завершения ---
    if wfo_results and wfo_results.get('summary'):
        st.success("WFO с автоматическим подбором диапазонов успешно завершен!")
        from visualizer import plot_wfo_results, plot_wfo_parameter_stability, plot_wfo_insample_vs_outsample
        summary_df = pd.DataFrame(wfo_results['summary'])
        st.dataframe(summary_df)
        st.plotly_chart(plot_wfo_results(summary_df, wfo_results['equity_curve'], wfo_results['aggregated_metrics']), use_container_width=True)
        st.plotly_chart(plot_wfo_parameter_stability(summary_df, new_param_space), use_container_width=True)
        st.plotly_chart(plot_wfo_insample_vs_outsample(summary_df), use_container_width=True)
    else:
        st.error("WFO с автоматическим подбором диапазонов не дал результатов.")