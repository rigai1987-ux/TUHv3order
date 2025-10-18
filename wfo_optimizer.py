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
from signal_generator import find_future_outcomes
import optuna_optimizer
from trading_simulator import run_trading_simulation
import streamlit as st
import numpy as np

def suggest_new_ranges_from_wfo(wfo_summary_df: pd.DataFrame, initial_param_space: dict) -> dict:
    """
    Анализирует результаты WFO и предлагает новые, суженные диапазоны параметров.

    Args:
        wfo_summary_df (pd.DataFrame): Сводная таблица по шагам WFO.
        initial_param_space (dict): Исходное пространство параметров, чтобы знать, какие из них были оптимизированы.

    Returns:
        dict: Словарь с предложенными диапазонами.
    """
    if wfo_summary_df.empty:
        return {}

    # 1. Отбираем только успешные шаги (где PnL на тесте > 0)
    successful_steps = wfo_summary_df[wfo_summary_df['out_sample_pnl'] > 0]

    if len(successful_steps) < 3: # Нужно хотя бы несколько точек для статистики
        st.warning("Недостаточно успешных шагов WFO для предложения новых диапазонов.")
        return {}

    suggested_ranges = {}
    # 2. Итерируемся по параметрам, которые были в оптимизации
    for param_name, (param_type, *_) in initial_param_space.items():
        if param_name in successful_steps.columns and param_type in ["int", "float"]:
            values = successful_steps[param_name].dropna()
            if len(values) > 1:
                # 3. Используем квантили для отсечения выбросов
                new_min = values.quantile(0.10)
                new_max = values.quantile(0.90)

                # Округляем для красивого вывода и избежания слишком длинных float
                suggested_ranges[param_name] = {
                    "min": round(new_min, 4) if param_type == "float" else int(new_min),
                    "max": round(new_max, 4) if param_type == "float" else int(new_max)
                }
    return suggested_ranges

def run_wfo(
    data: pd.DataFrame,
    wfo_params: dict,
    opt_params: dict,
    use_screening: bool = True,
    objective_name: str = "SQN", # Имя целевой метрики для отображения
    show_progress_ui: bool = True # Показывать ли UI на каждом шаге
):
    """
    Запускает полный цикл Walk-Forward Optimization.

    Args:
        data (pd.DataFrame): Полный набор данных для WFO.
        wfo_params (dict): Параметры для WFO (размеры окон, шаг).
            - train_period: Длительность окна обучения.
            - test_period: Длительность окна тестирования.
            - step_period: Шаг сдвига окна.
            - wfo_unit: Единица измерения ('Дни' или 'Часы').
        opt_params (dict): Параметры для Optuna-оптимизации, передаваемые в `run_optimization`.
        use_screening (bool): Если True, будет применяться предварительный скрининг сигналов на обучающих данных.
        objective_name (str): Название целевой метрики для корректного отображения в UI.
        show_progress_ui (bool): Если False, UI не будет обновляться на каждом шаге для ускорения.
    """
    train_period = wfo_params['train_period']
    test_period = wfo_params['test_period']
    step_period = wfo_params['step_period']
    wfo_unit = wfo_params.get('wfo_unit', 'Дни')

    start_date_data = data['datetime'].min()
    end_date_data = data['datetime'].max()

    if wfo_unit == 'Дни':
        time_unit = timedelta(days=1)
        total_duration = (end_date_data - start_date_data).days
        train_duration = timedelta(days=train_period)
        test_duration = timedelta(days=test_period)
        step_duration = timedelta(days=step_period)
    else: # Часы
        time_unit = timedelta(hours=1)
        total_duration = int((end_date_data - start_date_data).total_seconds() / 3600)
        train_duration = timedelta(hours=train_period)
        test_duration = timedelta(hours=test_period)
        step_duration = timedelta(hours=step_period)

    if total_duration < (train_period + test_period):
        st.error("Недостаточно данных для проведения WFO с заданными размерами окон.")
        return {"summary": [], "aggregated_metrics": {}, "equity_curve": pd.DataFrame()}

    # Инициализация для цикла WFO
    current_start_date = start_date_data
    out_of_sample_results = []
    wfo_summary = []
    all_oos_trades = [] # Собираем все сделки для построения кривой доходности
    previous_best_params = None # Хранилище для лучших параметров с предыдущего шага

    if show_progress_ui:
        pbar = st.progress(0)
    
    # Расчет общего количества шагов
    total_walk_duration = total_duration - train_period - test_period
    if total_walk_duration < 0:
        total_steps = 0
    else:
        total_steps = (total_walk_duration // step_period) + 1

    step_count = 0

    while current_start_date + train_duration + test_duration <= end_date_data:
        step_count += 1
        
        # 2. Определяем окна на основе дат
        train_start_date = current_start_date
        train_end_date = train_start_date + train_duration
        test_start_date = train_end_date
        test_end_date = test_start_date + test_duration

        if show_progress_ui:
            # Создаем контейнер для вывода информации о шаге и ключевых метрик
            step_container = st.container()
            with step_container:
                info_cols = st.columns([4, 1, 1]) # Колонки для текста, метрики и кол-ва сделок

        # Форматируем вывод дат для UI
        date_format = "%Y-%m-%d" if wfo_unit == "Дни" else "%Y-%m-%d %H:%M"
        train_period_str = f"{train_start_date.strftime(date_format)} : {train_end_date.strftime(date_format)}"
        test_period_str = f"{test_start_date.strftime(date_format)} : {test_end_date.strftime(date_format)}"

        # 3. Разделяем данные по датам
        # Создаем явные копии, чтобы избежать SettingWithCopyWarning
        train_data = data[(data['datetime'] >= train_start_date) & (data['datetime'] < train_end_date)].copy()
        test_data = data[(data['datetime'] >= test_start_date) & (data['datetime'] < test_end_date)].copy()

        if train_data.empty or test_data.empty:
            st.warning(f"Пропуск шага: недостаточно данных в периоде.")
            current_start_date += step_duration
            continue

        # 3.5. Опционально проводим скрининг "перспективных" сигналов на обучающих данных
        if use_screening:
            if show_progress_ui:
                st.write("🔬 **Включен режим предварительного скрининга.**")
            # Получаем параметры скрининга из wfo_params
            look_forward_period = wfo_params.get('look_forward_period', 20)
            profit_target_pct = wfo_params.get('profit_target_pct', 2.0)
            loss_limit_pct = wfo_params.get('loss_limit_pct', 1.0)
            
            promising_long, promising_short = find_future_outcomes(
                train_data['high'].values, train_data['low'].values,
                int(look_forward_period), profit_target_pct / 100, loss_limit_pct / 100
            )
            train_data['promising_long'] = promising_long
            train_data['promising_short'] = promising_short
            
            # Логируем результаты скрининга для прозрачности
            num_promising_long = np.sum(promising_long)
            num_promising_short = np.sum(promising_short)
            if show_progress_ui:
                st.write(f"🔍 Найдено {num_promising_long} перспективных long и {num_promising_short} перспективных short сигналов для обучения.")

        # 4. Запускаем оптимизацию на обучающем окне (In-Sample)
        opt_params['data'] = train_data
        # Уменьшаем количество проб для каждого шага WFO для ускорения
        opt_params['n_trials'] = wfo_params.get('trials_per_step', 25)
        # ВАЖНО: Передаем флаг скрининга в параметры оптимизации,
        # чтобы целевая функция знала, в каком режиме работать.
        opt_params['screening_mode_on_train'] = use_screening
        # Передаем порог по сделкам в базовые настройки, чтобы он попал в целевую функцию
        # --- Новая логика: передаем "анкерные" параметры с предыдущего шага ---
        if wfo_params.get('use_anchoring') and previous_best_params:
            opt_params['seed_params'] = previous_best_params

        if 'base_settings' not in opt_params:
            opt_params['base_settings'] = {}
        opt_params['base_settings']['min_trades_threshold'] = wfo_params.get('min_trades_threshold', 10)

        in_sample_opt_results = optuna_optimizer.run_optimization(opt_params)

        if not in_sample_opt_results or not in_sample_opt_results.get('best_params'):
            st.warning(f"На шаге {step_count} не найдено оптимальных параметров. Пропуск.")
            current_start_date += step_duration
            continue

        # --- Определение лучшей пробы и ее результатов для логирования ---
        is_multi_objective = "multi" in str(opt_params.get('direction', ''))
        best_trial = None
        study = in_sample_opt_results.get('study')

        if study:
            if is_multi_objective:
                # Для многоцелевой оптимизации выбираем лучший результат по первой метрике (SQN)
                best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
                if best_trials:
                    best_trial = best_trials[0]
            else:
                # Для одноцелевой оптимизации просто берем best_trial
                best_trial = study.best_trial

        best_params = best_trial.params if best_trial else in_sample_opt_results['best_params']
        best_value = best_trial.values if best_trial and is_multi_objective else (best_trial.value if best_trial else in_sample_opt_results['best_value'])

        # --- Новая логика: сохраняем лучшие параметры для следующего шага ---
        previous_best_params = best_params.copy()

        # Получаем реальные результаты лучшей пробы для логирования
        best_trial_sim_results = best_trial.user_attrs if best_trial else {}
        feature_importances = best_trial_sim_results.get('feature_importances')

        if show_progress_ui:
            # Выводим информацию о шаге и метрики в заранее созданный контейнер
            with step_container:
                with info_cols[0]:
                    st.info(f"**Шаг {step_count}/{total_steps}**: Обучение [{train_period_str}], Тест [{test_period_str}]")
                
                main_metric_label = objective_name.split('(')[0].strip()
                with info_cols[1]:
                    if is_multi_objective:
                        st.metric(label=f"Метрика ({main_metric_label})", value=f"{best_value[0]:.4f}", help=f"Лучшее значение целевой метрики на шаге {step_count} (In-Sample)")
                    else:
                        st.metric(label=f"Метрика ({main_metric_label})", value=f"{best_value:.4f}", help=f"Лучшее значение целевой метрики на шаге {step_count} (In-Sample)")
                with info_cols[2]:
                    st.metric(label="Сделок (In-Sample)", value=best_trial_sim_results.get('total_trades', 'N/A'), help=f"Количество сделок, найденное с лучшими параметрами на шаге {step_count}")

            # В спойлере оставляем только детальную информацию - найденные параметры
            with st.expander(f"⚙️ Найденные параметры на шаге {step_count}"):
                    st.json(best_params)
        else:
            # Выводим прогресс в консоль, если UI отключен
            print(f"WFO Шаг {step_count}/{total_steps} завершен.")

        # 5. Тестируем лучшие параметры на тестовом окне (Out-of-Sample)
        # Собираем все параметры: базовые настройки из opt_params и лучшие найденные параметры.
        # best_params будут иметь приоритет при совпадении ключей.
        simulation_params = {
            **opt_params.get('base_settings', {}),
            **best_params
        }
        
        out_of_sample_run = run_trading_simulation(test_data, simulation_params, screening_mode=False) # На тесте работаем в обычном режиме
        out_of_sample_results.append(out_of_sample_run)

        # Добавляем сделки из этого шага в общий список
        if out_of_sample_run['trades']:
            trades_df = pd.DataFrame(out_of_sample_run['trades'])
            # Важно: exit_idx относится к test_data, а не к глобальному data.
            # Поэтому exit_time нужно получать из test_data.
            trades_df['exit_time'] = test_data['datetime'].iloc[trades_df['exit_idx']].values
            all_oos_trades.append(trades_df)
        
        # 6. Сохраняем сводную информацию по шагу
        summary_step = {
            "step": step_count,
            "train_period": train_period_str,
            "test_period": test_period_str,
            "in_sample_metric": in_sample_opt_results['best_value'],
            "out_sample_pnl": out_of_sample_run['total_pnl'],
            "out_sample_trades": out_of_sample_run['total_trades'],
            "out_sample_win_rate": out_of_sample_run['win_rate'],
            "feature_importances": feature_importances, # Сохраняем важность признаков
        }
        # Раскладываем лучшие параметры по отдельным колонкам для удобства анализа
        summary_step.update(best_params)
        wfo_summary.append(summary_step)

        # 7. Сдвигаем окно
        current_start_date += step_duration
        if show_progress_ui:
            pbar.progress(step_count / total_steps if total_steps > 0 else 1.0)

    if show_progress_ui:
        pbar.progress(1.0)
    st.success("Walk-Forward оптимизация завершена!")

    # 8. Агрегируем и возвращаем результаты
    if not out_of_sample_results:
        return {"summary": [], "aggregated_metrics": {}, "equity_curve": pd.DataFrame()}

    # Создаем единый DataFrame для кривой доходности
    equity_curve_df = pd.DataFrame()
    if all_oos_trades:
        full_trades_df = pd.concat(all_oos_trades).sort_values('exit_time').reset_index(drop=True)
        full_trades_df['cumulative_pnl'] = full_trades_df['pnl'].cumsum()
        equity_curve_df = full_trades_df[['exit_time', 'cumulative_pnl']]

    # Расчет агрегированных метрик
    total_pnl = equity_curve_df['cumulative_pnl'].iloc[-1] if not equity_curve_df.empty else 0
    all_trades_count = sum([res['total_trades'] for res in out_of_sample_results])
    all_wins = sum([res['winning_trades'] for res in out_of_sample_results])
    
    all_pnl_history = [pnl for res in out_of_sample_results for pnl in res['pnl_history']]
    if all_pnl_history:
        profits = np.sum([p for p in all_pnl_history if p > 0])
        losses = np.abs(np.sum([p for p in all_pnl_history if p < 0]))
        profit_factor = profits / losses if losses > 0 else float('inf')
    else:
        profit_factor = 0

    aggregated_metrics = {
        "total_out_of_sample_pnl": total_pnl,
        "total_out_of_sample_trades": all_trades_count,
        "overall_win_rate": all_wins / all_trades_count if all_trades_count > 0 else 0,
        "overall_profit_factor": profit_factor
    }

    # 9. Предлагаем новые диапазоны на основе успешных шагов
    wfo_summary_df = pd.DataFrame(wfo_summary)
    suggested_ranges = suggest_new_ranges_from_wfo(wfo_summary_df, opt_params['param_space'])

    return {
        "summary": wfo_summary, "aggregated_metrics": aggregated_metrics, 
        "equity_curve": equity_curve_df, "suggested_ranges": suggested_ranges
    }

def _wfo_optimization_task(task_params):
    """
    Вспомогательная функция для выполнения одной задачи оптимизации в параллельном режиме.
    Принимает и возвращает словарь, чтобы быть совместимой с joblib.
    """
    step_count = task_params['step_count']
    total_steps = task_params['total_steps']
    opt_params = task_params['opt_params']
    opt_params['n_jobs'] = 1 # ВАЖНО: Каждая задача WFO выполняется в одном потоке
    
    print(f"Starting parallel WFO optimization for step {step_count}/{total_steps}...")
    in_sample_opt_results = optuna_optimizer.run_optimization(opt_params)
    print(f"Finished parallel WFO optimization for step {step_count}/{total_steps}.")
    
    task_params['in_sample_opt_results'] = in_sample_opt_results
    return task_params


def run_wfo_parallel(
    data: pd.DataFrame,
    wfo_params: dict,
    opt_params: dict,
    use_screening: bool = True
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

        train_data = data[(data['datetime'] >= train_start_date) & (data['datetime'] < train_end_date)].copy()
        test_data = data[(data['datetime'] >= test_start_date) & (data['datetime'] < test_end_date)].copy()

        if train_data.empty or test_data.empty:
            current_start_date += step_duration
            continue

        # Подготовка параметров для конкретной задачи
        task_opt_params = opt_params.copy()
        task_opt_params['data'] = train_data
        task_opt_params['n_trials'] = wfo_params.get('trials_per_step', 25)
        task_opt_params['screening_mode_on_train'] = use_screening
        if 'base_settings' not in task_opt_params: task_opt_params['base_settings'] = {}
        task_opt_params['base_settings']['min_trades_threshold'] = wfo_params.get('min_trades_threshold', 10)

        tasks.append({
            "step_count": i + 1,
            "total_steps": total_steps,
            "train_data": train_data,
            "test_data": test_data,
            "opt_params": task_opt_params,
            "wfo_params": wfo_params,
            "use_screening": use_screening,
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

    for task in sorted(completed_tasks, key=lambda x: x['step_count']):
        in_sample_opt_results = task['in_sample_opt_results']
        
        if not in_sample_opt_results or not in_sample_opt_results.get('best_params'):
            st.warning(f"На шаге {task['step_count']} не найдено оптимальных параметров. Пропуск.")
            continue

        best_params = in_sample_opt_results['best_params']

        # --- Извлечение feature_importances из лучшей пробы ---
        feature_importances = None
        study = in_sample_opt_results.get('study')
        if study and study.best_trial:
            feature_importances = study.best_trial.user_attrs.get('feature_importances')
        # --- Конец блока извлечения ---

        simulation_params = {**task['opt_params'].get('base_settings', {}), **best_params}
        
        out_of_sample_run = run_trading_simulation(task['test_data'], simulation_params, screening_mode=False)
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
            "feature_importances": feature_importances, # Добавляем важность признаков
        }
        summary_step.update(best_params)
        wfo_summary.append(summary_step)

    st.success("Walk-Forward оптимизация (параллельный режим) завершена!")

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
    profit_factor = np.sum([p for p in all_pnl_history if p > 0]) / np.abs(np.sum([p for p in all_pnl_history if p < 0])) if any(p < 0 for p in all_pnl_history) else float('inf')

    aggregated_metrics = {"total_out_of_sample_pnl": total_pnl, "total_out_of_sample_trades": all_trades_count, "overall_win_rate": all_wins / all_trades_count if all_trades_count > 0 else 0, "overall_profit_factor": profit_factor}
    wfo_summary_df = pd.DataFrame(wfo_summary)
    suggested_ranges = suggest_new_ranges_from_wfo(wfo_summary_df, opt_params['param_space'])

    return {"summary": wfo_summary, "aggregated_metrics": aggregated_metrics, "equity_curve": equity_curve_df, "suggested_ranges": suggested_ranges}