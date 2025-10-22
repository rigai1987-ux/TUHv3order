import optuna
import numpy as np
import os, joblib
from joblib import parallel_backend
from joblib.parallel import ThreadingBackend
import streamlit as st

# Устанавливаем уровень логирования Optuna. INFO - для отображения проб. WARNING - для "тихого" режима.
optuna.logging.set_verbosity(optuna.logging.INFO)

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
    backend_choice = params.get('backend_choice', 'loky') # По умолчанию 'loky'

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
            return suggested_params

        trial_params = suggest_params_recursively(param_space, trial)

        # Объединяем параметры из trial с базовыми настройками, переданными из UI
        # Базовые настройки (position_size, commission и т.д.) теперь передаются явно
        full_params = {**params.get('base_settings', {}), **trial_params}

        # Запускаем целевую функцию
        try:
            # strategy_func теперь может возвращать кортеж (метрики, результаты_симуляции)
            result = strategy_func(data, full_params)
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
            # Использование TrialPruned - более идиоматичный способ сообщить Optuna о неудаче.
            # Это позволяет Optuna лучше обрабатывать такие случаи, например, в TPE-сэмплере.
            # return (-1000.0, -1.0) if is_multi_objective else -1000.0
            raise optuna.exceptions.TrialPruned(f"Ошибка в пробе: {str(e)}")

        # Проверка на достижение цели
        if target_metric_value is not None:
            current_value = metric[0] if isinstance(metric, tuple) else metric
            if current_value >= target_metric_value:
                trial.study.stop()
        
        # Если метрика является штрафом, сохраняем причину для отладки
        if sim_results: # sim_results будет пустым, если была ошибка
            if (is_multi_objective and metric[0] < 0) or (not is_multi_objective and metric < 0):
                # --- ИСПРАВЛЕНИЕ: Безопасное форматирование PnL ---
                pnl_value = sim_results.get('total_pnl', 'N/A')
                pnl_str = f"{pnl_value:.2f}" if isinstance(pnl_value, (int, float)) else str(pnl_value)
                trial.set_user_attr("reason_for_penalty", f"Недостаточно сделок или убыточность. Trades: {sim_results.get('total_trades', 'N/A')}, PnL: {pnl_str}")

        return metric

    is_multi_objective = "multi" in str(params.get('direction', ''))
    direction = "maximize" if not is_multi_objective else ["maximize", "maximize", "maximize"]

    # Определяем количество параллельных задач.
    # Используем все доступные логические ядра для максимальной загрузки ЦП.
    # os.cpu_count() возвращает количество логических ядер.
    # Если n_jobs передано извне (например, из WFO), используем это значение.
    # В противном случае, используем все ядра.
    n_jobs = params.get('n_jobs', os.cpu_count() or 1)

    # Заменяем st.info на print, чтобы избежать вызова UI-функций из потенциально фоновых потоков (например, в WFO).
    # Основной UI (спиннер) будет отображаться на странице optimization_page.
    print(f"Запуск Optuna оптимизации с {n_trials} пробами на {n_jobs} ядрах. Бэкенд: {backend_choice}")

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
            print(f"Warning: Не удалось добавить анкерную пробу: {e}")

    try:
        # Используем 'loky' по умолчанию. Он более надежен в среде Streamlit, так как использует процессы.
        with parallel_backend('loky'):
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    except optuna.exceptions.TrialPruned:
        print("Оптимизация была остановлена (возможно, пользователем или из-за ошибки в пробе).")

    if os.path.exists(stop_file):
        os.remove(stop_file)

    # После завершения оптимизации выводим все собранные ошибки в основном потоке
    # --- ИСПРАВЛЕНИЕ: Не вызываем st.error из этой функции.
    # Вместо этого, возвращаем ошибки, чтобы вызывающий код мог их обработать.
    # Это предотвращает ошибку 'missing ScriptRunContext' при вызове из WFO.
    
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
                # --- ИСПРАВЛЕНИЕ: Безопасное форматирование PnL ---
                if 'pnl' in key.lower() and isinstance(value, (int, float)):
                    result_entry[key] = f"${value:.2f}"
                else:
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
                    # --- ИСПРАВЛЕНИЕ: Безопасное форматирование PnL ---
                    if 'pnl' in key.lower() and isinstance(value, (int, float)):
                        result_entry[key] = f"${value:.2f}"
                    else:
                        result_entry[key] = value
                top_10_results.append(result_entry)

    return {
        'best_value': best_value,
        'best_params': best_params,
        'top_10_results': top_10_results,        
        'study': study,
        'errors': errors # Возвращаем список ошибок
    }

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

    # --- НОВЫЙ БЛОК: Отображение ошибок после завершения оптимизации ---
    # Так как run_iterative_optimization всегда вызывается из основного потока,
    # здесь безопасно использовать st.error.
    if final_results and final_results.get('errors'):
        st.error(f"Во время оптимизации произошло {len(final_results['errors'])} ошибок в пробах:")
        for error_info in final_results['errors'][:5]: # Показываем первые 5
            with st.expander(f"Ошибка в пробе #{error_info['trial_number']}: {error_info['error']}"):
                st.json(error_info['params'])

    return final_results