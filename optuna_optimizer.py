import optuna
import pandas as pd
import pprint
import threading
import numpy as np
from datetime import datetime, date
import os, joblib
from joblib import parallel_backend
from joblib.parallel import ThreadingBackend
import streamlit as st

# Устанавливаем уровень логирования Optuna. INFO - для отображения проб. WARNING - для "тихого" режима.
optuna.logging.set_verbosity(optuna.logging.INFO)


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def run_optimization(params):
    """
    Запускает оптимизацию с использованием Optuna.
    """
    data = params['data']
    param_space = params['param_space']
    n_trials = params['n_trials']
    strategy_func = params['strategy_func']
    seed_params = params.get('seed_params') # Получаем "анкерные" параметры
    target_metric_value = params.get('target_metric_value')
    backend_choice = params.get('backend_choice', 'threading') # По умолчанию 'threading'

    # Флаг для остановки
    stop_file = 'stop_optimization.flag'
    if os.path.exists(stop_file):
        os.remove(stop_file)

    # Потокобезопасный список для сбора ошибок из разных потоков
    errors = []

    def objective(trial):
        if os.path.exists(stop_file):
            trial.study.stop()
            raise optuna.exceptions.TrialPruned("Оптимизация остановлена пользователем.")

        def suggest_params_recursively(current_param_space, current_trial):
            """Рекурсивно предлагает параметры, поддерживая условные зависимости."""
            suggested_params = {}
            for name, value_tuple in current_param_space.items():
                param_type = value_tuple[0]
                if param_type == "int":
                    _, low, high = value_tuple
                    suggested_params[name] = current_trial.suggest_int(name, low, high)
                elif param_type == "float":
                    _, low, high = value_tuple
                    suggested_params[name] = current_trial.suggest_float(name, low, high)
                elif param_type == "categorical":
                    choices = value_tuple[1]
                    suggested_params[name] = current_trial.suggest_categorical(name, choices)
                elif param_type == "conditional":
                    # value_tuple: ("conditional", conditional_param_name, conditional_param_space)
                    _, conditional_param_name, conditional_param_space = value_tuple
                    # Убедимся, что условный параметр уже был предложен
                    if conditional_param_name in suggested_params:
                        chosen_value = suggested_params[conditional_param_name]
                        # Если для выбранного значения есть свое пространство параметров, предлагаем их
                        if chosen_value in conditional_param_space:
                            nested_params = suggest_params_recursively(conditional_param_space[chosen_value], current_trial)
                            suggested_params.update(nested_params)
                    else:
                        # Это может произойти, если 'conditional' определен до 'categorical' в словаре
                        # В Python 3.7+ порядок ключей словаря сохраняется, но лучше располагать 'categorical' раньше.
                        st.warning(f"Условный параметр '{name}' определен до его зависимости '{conditional_param_name}'. "
                                   f"Убедитесь, что зависимость '{conditional_param_name}' идет первой в param_space.")
            return suggested_params

        trial_params = suggest_params_recursively(param_space, trial)

        # Объединяем параметры из trial с базовыми настройками, переданными из UI
        # Базовые настройки (position_size, commission и т.д.) теперь передаются явно
        full_params = {**params.get('base_settings', {}), **trial_params}

        # print(f"\n--- Optuna Trial #{trial.number} с параметрами: ---")
        # pprint.pprint(full_params)
        # print("---------------------------------------------------\n")
 
        # Запускаем целевую функцию
        try:
            # strategy_func теперь может возвращать кортеж (метрики, результаты_симуляции)
            # Проверяем, есть ли в данных колонки для скрининга. Если да, включаем screening_mode.
            # Используем флаг, переданный из WFO, чтобы контролировать, когда включать скрининг.
            use_screening_for_this_run = params.get('screening_mode_on_train', False) and 'promising_long' in data.columns
            result = strategy_func(data, full_params, screening_mode=use_screening_for_this_run)
            sim_results = {}
            if isinstance(result, tuple) and len(result) == 2:
                metric, sim_results = result
                # Сохраняем дополнительные метрики в атрибуты пробы
                if isinstance(sim_results, dict):
                    for key, value in sim_results.items():
                        # Исключаем большие объекты, которые не нужны в логах Optuna, но сохраняем feature_importances
                        if key not in ['trades', 'pnl_history', 'balance_history', 'used_params']:
                            trial.set_user_attr(key, value)
                # Для одноцелевой оптимизации, если возвращается кортеж, берем первый элемент
                if not is_multi_objective and isinstance(metric, (list, tuple)):
                    metric = metric[0]
            else:
                metric = result # Для обратной совместимости
        except Exception as e:
            # НЕ используем st.error здесь. Собираем ошибки в потокобезопасный список.
            errors.append({
                "trial_number": trial.number,
                "params": trial_params,
                "error": str(e)
            })
            # Возвращаем худшее значение, чтобы Optuna знала, что это плохой результат
            return (-1000.0, -1.0) if is_multi_objective else -1000.0

        # Проверка на достижение цели
        if target_metric_value is not None:
            current_value = metric[0] if isinstance(metric, tuple) else metric
            if current_value >= target_metric_value:
                trial.study.stop()
        
        # Если метрика является штрафом, сохраняем причину для отладки
        if sim_results: # sim_results будет пустым, если была ошибка
            if (is_multi_objective and metric[0] < 0) or (not is_multi_objective and metric < 0):
                trial.set_user_attr("reason_for_penalty", f"Недостаточно сделок или убыточность. Trades: {sim_results.get('total_trades', 'N/A')}, PnL: {sim_results.get('total_pnl', 'N/A'):.2f}")

        return metric

    is_multi_objective = "multi" in str(params.get('direction', ''))
    direction = "maximize" if not is_multi_objective else ["maximize", "maximize", "maximize"]

    # Определяем количество параллельных задач.
    # Используем все доступные логические ядра для максимальной загрузки ЦП.
    # os.cpu_count() возвращает количество логических ядер.
    # Если n_jobs передано извне (например, из WFO), используем это значение.
    # В противном случае, используем все ядра.
    n_jobs = params.get('n_jobs', os.cpu_count() or 1)

    st.info(f"Запуск Optuna оптимизации с {n_trials} пробами на {n_jobs} ядрах. Это может занять некоторое время...")
    st.info(f"Используется бэкенд для параллелизма: **{backend_choice}**")
    study = optuna.create_study(directions=direction if is_multi_objective else [direction])

    # --- Новая логика: добавляем "анкерную" пробу, если она предоставлена ---
    if seed_params:
        try:
            # Убедимся, что передаем только те параметры, которые есть в текущем пространстве поиска
            # Это важно, т.к. пространство может меняться (например, при сужении диапазонов)
            valid_seed_params = {k: v for k, v in seed_params.items() if k in param_space}
            study.enqueue_trial(valid_seed_params)
            # Не выводим сообщение в Streamlit, чтобы не засорять лог WFO
            print(f"Info: Enqueued seed trial for Optuna study with params: {valid_seed_params}")
        except Exception as e:
            st.warning(f"Не удалось добавить анкерную пробу: {e}")

    try:
        if backend_choice == 'threading':
            # 'threading' имеет меньшие накладные расходы, идеально для быстрых вычислений с Numba.
            # Явное создание экземпляра решает проблему с замедлением при повторных запусках в Streamlit,
            # гарантируя создание нового, чистого пула потоков для каждого вызова.
            backend = ThreadingBackend(inner_max_num_threads=n_jobs)
        else: # 'loky'
            # 'loky' использует процессы, что обеспечивает лучшую изоляцию и надежность,
            # но имеет более высокие накладные расходы.
            # Для 'loky' не требуется специальная обработка, как для ThreadingBackend.
            backend = 'loky'

        with parallel_backend(backend):
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    except optuna.exceptions.TrialPruned:
        st.warning("Оптимизация была остановлена.")

    if os.path.exists(stop_file):
        os.remove(stop_file)

    # После завершения оптимизации выводим все собранные ошибки в основном потоке
    if errors:
        st.error(f"Во время оптимизации произошло {len(errors)} ошибок в пробах:")
        for error_info in errors[:5]: # Показываем первые 5 ошибок, чтобы не перегружать интерфейс
            with st.expander(f"Ошибка в пробе #{error_info['trial_number']}: {error_info['error']}"):
                st.json(error_info['params'])

    if is_multi_objective:
        best_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
        if not best_trials:
            return {'best_value': None, 'best_params': {}, 'top_10_results': []}
        
        best_trial = best_trials[0]
        best_value = best_trial.values
        best_params = best_trial.params

        top_10_results = []
        for t in best_trials[:10]:
            result_entry = t.params.copy()
            result_entry['trial_number'] = t.number
            result_entry['value'] = t.values
            # Добавляем сохраненные метрики
            for key, value in t.user_attrs.items():
                # Форматируем PnL для лучшего отображения
                if 'pnl' in key.lower() and isinstance(value, (int, float)):
                    result_entry[key] = f"${value:.2f}"
                result_entry[key] = value
            top_10_results.append(result_entry)

    else: # Single objective
        if not study.best_trials:
            return {'best_value': None, 'best_params': {}, 'top_10_results': []}

        best_trial = study.best_trial
        best_value = best_trial.value
        best_params = best_trial.params

        top_10_results = []
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)
        for t in sorted_trials[:10]:
            if t.value is not None:
                result_entry = t.params.copy()
                result_entry['trial_number'] = t.number
                result_entry['value'] = t.value
                # Добавляем сохраненные метрики
                for key, value in t.user_attrs.items():
                    # Форматируем PnL для лучшего отображения
                    if 'pnl' in key.lower() and isinstance(value, (int, float)):
                        result_entry[key] = f"${value:.2f}"
                    result_entry[key] = value
                top_10_results.append(result_entry)

    return {
        'best_value': best_value,
        'best_params': best_params,
        'top_10_results': top_10_results,        
        'study': study,
    }

def _flatten_conditional_params(params: dict) -> dict:
    """
    Упрощает словарь параметров, "схлопывая" условные параметры в одну колонку.
    Например: {'classifier_type': 'RandomForest', 'rf_n_estimators': 100}
    превращается в {'classifier_type': 'RandomForest', 'classifier_details': {'n_estimators': 100}}
    """
    classifier_type = params.get('classifier_type')
    if not classifier_type:
        return params

    # Определяем префикс для поиска параметров. CatBoost не имеет верхнего регистра в названии.
    prefix = classifier_type.lower() + '_'

    details = {k.replace(prefix, ''): v for k, v in params.items() if k.startswith(prefix)}

    if not details:
        return params

    # Удаляем исходные ключи и добавляем новый
    params_to_remove = [k for k in params if k.startswith(prefix)]
    for k in params_to_remove:
        del params[k]
    params['classifier_details'] = str(details) # Преобразуем в строку для DataFrame
    return params

def run_iterative_optimization(params):
    """
    Запускает итеративную оптимизацию.
    """
    st.info("Этап 1: Широкий поиск...")
    initial_results = run_optimization(params)
    
    if not initial_results or not initial_results.get('best_params'):
        st.error("Начальный этап оптимизации не дал результатов. Проверьте диапазоны.")
        return None

    st.info("Этап 2: Углубленный поиск вокруг лучших параметров...")
    
    best_params = initial_results['best_params']
    new_param_space = {}
    
    for name, value_tuple in params['param_space'].items():
        param_type = value_tuple[0]
        
        # Пропускаем условные "псевдо-параметры", так как они не имеют собственных диапазонов
        if param_type == "conditional":
            new_param_space[name] = value_tuple
            continue

        # Для категориальных параметров оставляем их как есть, не сужая выбор
        if param_type == "categorical":
            new_param_space[name] = value_tuple
            continue

        # Обработка числовых параметров (int, float)
        if name not in best_params:
            # Если по какой-то причине параметра нет в лучших (например, из-за условной логики),
            # оставляем исходный диапазон
            new_param_space[name] = value_tuple
            continue

        _, low, high = value_tuple
        best_val = best_params[name]

        if param_type == "int":
            spread = max(1, (high - low) // 4)
            new_low = max(low, best_val - spread)
            new_high = min(high, best_val + spread)
            # Убедимся, что диапазон не нулевой и корректен
            if new_low >= new_high:
                new_high = min(high, new_low + 1)
            new_param_space[name] = ("int", int(new_low), int(new_high))
        elif param_type == "float":
            spread = (high - low) / 4.0
            new_low = max(low, best_val - spread)
            new_high = min(high, best_val + spread)
            # Убедимся, что диапазон не нулевой
            if np.isclose(new_low, new_high):
                spread_fallback = (high - low) * 0.05 if high > low else 0.01
                new_high = min(high, new_high + spread_fallback)
                new_low = max(low, new_low - spread_fallback)
            new_param_space[name] = ("float", new_low, new_high)
        else:
            # На случай других типов параметров в будущем
            new_param_space[name] = value_tuple
            
    params['param_space'] = new_param_space
    
    final_results = run_optimization(params)

    # Улучшаем читаемость результатов, если они есть
    if final_results and 'top_10_results' in final_results:
        final_results['top_10_results'] = [_flatten_conditional_params(res) for res in final_results['top_10_results']]
        if 'best_params' in final_results:
            # Не изменяем best_params, так как они нужны для повторного запуска
            pass

    return final_results

def objective_wrapper(data, params, screening_mode=False):
    """
    Обертка для целевой функции, которая передает `screening_mode` в симулятор.
    """
    # Запускаем симуляцию с учетом режима скрининга
    results = run_trading_simulation(data, params, screening_mode=screening_mode)

    # Далее логика из `trading_strategy_objective_sqn`
    total_trades = results.get('total_trades', 0)
    if total_trades < 25: # min_trades_threshold
        return -100.0 + total_trades, results

    # ... (остальная логика расчета SQN, как в strategy_objectives.py)
    # Этот пример упрощен. В реальном коде нужно будет передавать `screening_mode`
    # через все слои до `run_trading_simulation`.
    # В нашем случае, мы уже изменили `run_optimization` для этого.
    pass