import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import json
import optuna
import plotly.express as px
# Импортируем необходимые функции из других модулей
from app_utils import get_basic_settings, manage_profiles, get_optimization_parameters, load_and_validate_data_files, _atomic_write
import wfo_optimizer
import optuna_optimizer
import strategy_objectives 
import visualizer # Импортируем модуль целиком


def get_param_space_from_ui():
    """Собирает пространство параметров для Optuna из session_state."""
    
    param_space = {}
    
    # Определение параметров и их диапазонов
    param_definitions = {
        "vol_period": ("int", "vol_period_min_optimization", "vol_period_max_optimization"),
        "vol_pctl": ("int", "vol_pctl_min_optimization", "vol_pctl_max_optimization"),
        "range_period": ("int", "range_period_min_optimization", "range_period_max_optimization"),
        "rng_pctl": ("int", "rng_pctl_min_optimization", "rng_pctl_max_optimization"),
        "natr_period": ("int", "natr_period_min_optimization", "natr_period_max_optimization"),
        "natr_min": ("float", "natr_min_min_optimization", "natr_min_max_optimization"),
        "lookback_period": ("int", "lookback_period_min_optimization", "lookback_period_max_optimization"),
        "min_growth_pct": ("float", "min_growth_pct_min_optimization", "min_growth_pct_max_optimization"),
        "stop_loss_pct": ("float", "stop_loss_pct_min_optimization", "stop_loss_pct_max_optimization"),
        "take_profit_pct": ("float", "take_profit_pct_min_optimization", "take_profit_pct_max_optimization"),
        "ml_iterations": ("int", "ml_iterations_min_optimization", "ml_iterations_max_optimization"),
        "ml_depth": ("int", "ml_depth_min_optimization", "ml_depth_max_optimization"),
        # --- НОВОЕ: Добавляем параметры для MLP ---
        "ml_epochs": ("int", "ml_epochs_min_optimization", "ml_epochs_max_optimization"),
        "ml_hidden_size": ("int", "ml_hidden_size_min_optimization", "ml_hidden_size_max_optimization"),
        "ml_num_hidden_layers": ("int", "ml_num_hidden_layers_min_optimization", "ml_num_hidden_layers_max_optimization"),
        "ml_batch_size": ("int", "ml_batch_size_min_optimization", "ml_batch_size_max_optimization"),
        "ml_dropout_rate": ("float", "ml_dropout_rate_min_optimization", "ml_dropout_rate_max_optimization"),
        "ml_learning_rate": ("float", "ml_learning_rate_min_optimization", "ml_learning_rate_max_optimization"),
        "ml_prints_window": ("int", "ml_prints_window_min_optimization", "ml_prints_window_max_optimization"),
        "ml_labeling_timeout_candles": ("int", "ml_labeling_timeout_candles_min_optimization", "ml_labeling_timeout_candles_max_optimization"),
    }

    # entry_logic_mode = st.session_state.get("entry_logic_mode_optimization", "Принты и HLdir") # Логика входа теперь одна
    for name, (ptype, min_key, max_key) in param_definitions.items():
        min_val = st.session_state.get(min_key)
        max_val = st.session_state.get(max_key)
        if min_val is not None and max_val is not None:
            param_space[name] = (ptype, min_val, max_val)

    # Обработка логики входа

    # Обработка параметров "вилки"
    param_space["bracket_offset_pct"] = ("float", st.session_state.get("bracket_offset_pct_min_optimization"), st.session_state.get("bracket_offset_pct_max_optimization"))
    param_space["bracket_timeout_candles"] = ("int", st.session_state.get("bracket_timeout_candles_min_optimization"), st.session_state.get("bracket_timeout_candles_max_optimization"))

    # Удаляем None значения, если виджеты не были созданы
    return {k: v for k, v in param_space.items() if all(i is not None for i in v[1:])}

def _save_optimization_run(run_type, results, selected_files, base_settings, start_date, end_date):
    """Вспомогательная функция для сохранения результатов оптимизации."""
    if not results or not results.get('best_params'):
        st.warning("Нет результатов для сохранения.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if len(selected_files) == 1:
        filename = selected_files[0]
        dash_pos = filename.find('-')
        data_prefix = filename[:dash_pos] if dash_pos != -1 else filename.rsplit('.', 1)[0]
    elif len(selected_files) > 1:
        data_prefix = "ALL"
    else:
        data_prefix = ""

    def extract_numeric_value(value):
        if value is None or (isinstance(value, float) and np.isnan(value)): return 0
        if isinstance(value, (list, tuple)): value = value[0]
        if hasattr(value, 'item'): value = value.item()
        numeric_str = ''.join(filter(lambda x: x.isdigit() or x in '.-', str(value)))
        try: return int(float(numeric_str) + 0.5)
        except (ValueError, TypeError): return 0

    best_value_num = extract_numeric_value(results.get('best_value'))
    run_name = f"run_{timestamp}_{data_prefix}_{run_type}_${best_value_num}_OPTUNA"
    
    ranges_dict = {k: v for k, v in st.session_state.items() if k.endswith('_optimization')}
    full_settings = {**base_settings, "start_date": str(start_date), "end_date": str(end_date)}
    
    run_data = {
        "run_name": run_name, "timestamp": datetime.now().isoformat(),
        "ranges": ranges_dict, "settings": full_settings, "data_files": selected_files,
        "best_params": results.get('best_params', {}), "optimization_type": f"optuna_{run_type.lower()}",
        "top_10_results": results.get('top_10_results', [])
    }

    try:
        os.makedirs("optimization_runs", exist_ok=True)
        file_path = os.path.join("optimization_runs", f"{run_name}.json")
        json_data = json.dumps(run_data, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else str(o))
        _atomic_write(file_path, json_data)
        st.success(f"Результаты оптимизации сохранены как '{run_name}'")
    except Exception as e:
        st.error(f"Ошибка при сохранении результатов: {str(e)}")

def show_optimization_page():
    """
    Отображает страницу "Оптимизация".
    """
    # Отображаем выбранные файлы в строке
    selected_files = st.session_state.get("selected_files_optimization", [])
    if selected_files:
        st.write(f"Выбраны файлы: {', '.join(selected_files)}")
    else:
        st.write("Файлы не выбраны")
    
    # Базовые настройки в боковой панели
    with st.sidebar:
        st.header("Настройки оптимизации")
        position_size, commission, start_date, end_date = get_basic_settings("optimization")

        manage_profiles("optimization", None) # params_func не используется для оптимизации
    
        st.subheader("Параметры оптимизации")
        
        st.info("Используется стандартная оптимизация **Optuna**.")
        optimization_type = "optuna"

        st.markdown("##### Логика входа: **Вилка отложенных ордеров**")
        st.session_state["entry_logic_mode_optimization"] = "Вилка отложенных ордеров"

        st.markdown("**Параметры вилки**")
        min_cols, max_cols = st.columns(2), st.columns(2)
        with min_cols[0]:
            st.number_input(
                "Отступ вилки (min, %)",
                value=float(st.session_state.get("bracket_offset_pct_min_optimization", 0.2)),
                min_value=0.01, step=0.01, key="bracket_offset_pct_min_optimization", format="%.2f"
            )
        with min_cols[1]:
            st.number_input(
                "Тайм-аут (min, свечи)",
                value=int(st.session_state.get("bracket_timeout_candles_min_optimization", 1)),
                min_value=1, step=1, key="bracket_timeout_candles_min_optimization"
            )
        with max_cols[0]:
            st.number_input(
                "Отступ вилки (max, %)",
                value=float(st.session_state.get("bracket_offset_pct_max_optimization", 1.0)),
                min_value=0.01, step=0.01, key="bracket_offset_pct_max_optimization", format="%.2f"
            )
        with max_cols[1]:
            st.number_input(
                "Тайм-аут (max, свечи)",
                value=int(st.session_state.get("bracket_timeout_candles_max_optimization", 10)),
                min_value=1, step=1, key="bracket_timeout_candles_max_optimization"
            )

        # Динамическое создание виджетов для диапазонов
        param_groups = {
            "Фильтр объёма": [("vol_period", "int"), ("vol_pctl", "int")],
            "Фильтр диапазона": [("range_period", "int"), ("rng_pctl", "int")],
            "Фильтр NATR": [("natr_period", "int"), ("natr_min", "float")],
            "Фильтр роста": [("lookback_period", "int"), ("min_growth_pct", "float")],
            "Управление риском": [("stop_loss_pct", "float"), ("take_profit_pct", "float")],
        }

        for group_name, params_in_group in param_groups.items():
            st.markdown(f"**{group_name}**")
            min_cols = st.columns(len(params_in_group))
            max_cols = st.columns(len(params_in_group))
            
            for i, (param_name, p_type) in enumerate(params_in_group):
                step = 0.01 if p_type == "float" else 1
                base_min_val = 1 if "period" in param_name or "window" in param_name else 0
                min_value_arg = float(base_min_val) if p_type == "float" else int(base_min_val)
                
                with min_cols[i]:
                    min_key = f"{param_name}_min_optimization"
                    min_val = st.session_state.get(min_key, min_value_arg)
                    min_val = float(min_val) if p_type == "float" else int(min_val)
                    st.number_input(f"{param_name} (min)", key=min_key, value=min_val, step=step, min_value=min_value_arg)
                
                with max_cols[i]:
                    max_key = f"{param_name}_max_optimization"
                    max_val = st.session_state.get(max_key, min_value_arg)
                    max_val = float(max_val) if p_type == "float" else int(max_val)
                    st.number_input(f"{param_name} (max)", key=max_key, value=max_val, step=step, min_value=min_value_arg)
        
        # --- НОВЫЙ БЛОК: Параметры для ML-модели ---
        # Эти виджеты будут отображаться всегда, но использоваться только если выбрана ML-цель
        st.markdown("**🤖 Параметры ML-модели**")

        # --- НОВЫЙ БЛОК: Выбор типа ML-модели ---
        ml_model_type_opt = st.radio(
            "Тип ML-модели для оптимизации",
            ["CatBoost", "NeuralNetwork"],
            key="ml_model_type_optimization",
            horizontal=True,
            help="Выберите модель, которая будет обучаться и оптимизироваться на каждом шаге WFO или в каждой пробе Optuna."
        )
        # --- Конец нового блока ---

        if ml_model_type_opt == "CatBoost":
            ml_param_groups = {
                "Гиперпараметры CatBoost": [("ml_iterations", "int"), ("ml_depth", "int"), ("ml_learning_rate", "float")],
            }
        elif ml_model_type_opt == "NeuralNetwork":
            ml_param_groups = {
                "Гиперпараметры Нейросети": [("ml_epochs", "int"), ("ml_hidden_size", "int"), ("ml_num_hidden_layers", "int"), ("ml_batch_size", "int"), ("ml_dropout_rate", "float"), ("ml_learning_rate", "float")],
            }
        else:
            ml_param_groups = {}

        # Общие параметры для ML
        ml_param_groups["Параметры признаков и разметки"] = [("ml_prints_window", "int"), ("ml_labeling_timeout_candles", "int")]

        for group_name, params_in_group in ml_param_groups.items():
            st.markdown(f"**{group_name}**")
            min_cols = st.columns(len(params_in_group))
            max_cols = st.columns(len(params_in_group))
            
            for i, (param_name, p_type) in enumerate(params_in_group):
                # --- ИСПРАВЛЕНИЕ: Устанавливаем шаг 0.1 для dropout_rate ---
                if "dropout" in param_name:
                    step = 0.1
                elif "learning_rate" in param_name:
                    step = 0.01
                else:
                    step = 10 if "iterations" in param_name else 8 if "size" in param_name or "batch" in param_name else 1

                # --- ИЗМЕНЕНИЕ: Устанавливаем минимальное значение для learning_rate ---
                if "iterations" in param_name:
                    base_min_val = 10
                elif "depth" in param_name:
                    base_min_val = 2
                elif "learning_rate" in param_name:
                    base_min_val = 0.0001 if ml_model_type_opt == "NeuralNetwork" else 0.01
                elif "dropout_rate" in param_name: # --- ИЗМЕНЕНИЕ: Минимальный dropout теперь 0.01 ---
                    base_min_val = 0.01
                elif "epochs" in param_name:
                    base_min_val = 5 # --- ИЗМЕНЕНИЕ: Явно устанавливаем минимум 5 ---
                elif "hidden_size" in param_name or "batch_size" in param_name:
                    base_min_val = 16
                else:
                    base_min_val = 1
                min_value_arg = float(base_min_val) if p_type == "float" else int(base_min_val)
                
                with min_cols[i]:
                    min_key = f"{param_name}_min_optimization"
                    st.number_input(f"{param_name} (min)", key=min_key, value=st.session_state.get(min_key, min_value_arg), step=step, min_value=min_value_arg, format="%.4f" if "learning_rate" in param_name and ml_model_type_opt == "NeuralNetwork" else "%.2f" if p_type == "float" else "%d")
                
                with max_cols[i]:
                    max_key = f"{param_name}_max_optimization"
                    st.number_input(f"{param_name} (max)", key=max_key, value=st.session_state.get(max_key, min_value_arg * 2), step=step, min_value=min_value_arg, format="%.4f" if "learning_rate" in param_name and ml_model_type_opt == "NeuralNetwork" else "%.2f" if p_type == "float" else "%d")


        col1, col2 = st.columns(2)
        with col1:
            optuna_trials = st.number_input("Количество проб Optuna", value=50, min_value=10, step=10, key="optuna_trials")
        with col2:
            wfo_trials = st.number_input("Количество проб WFO", value=20, min_value=5, step=5, key="wfo_trials")
        
        target_metric_value = st.number_input(
            "Остановить при достижении цели",
            value=None, placeholder="Не задано",
            help="Остановить оптимизацию, если значение целевой функции достигнет этого уровня."
        )

        objective_choice = st.selectbox(
            "Цель оптимизации",
            options=["SQN (стабильность)", "Плавность Equity (R-квадрат)", "HFT Score (частота и стабильность)", "Качество данных для ML (для WFO)", "SQN, Max Drawdown и Эффективность сигналов (многоцелевая)", "SQN с ML-фильтром (Optuna)", "SQN, Max Drawdown, Эффективность + ML (многоцелевая)"],
            index=2, key="objective_choice",
            options=["Calmar Ratio (риск/доходность)", "SQN (стабильность)", "Плавность Equity (R-квадрат)", "HFT Score (частота и стабильность)", "Качество данных для ML (для WFO)", "Прибыль, Просадка, Профит-фактор (многоцелевая)", "Calmar Ratio с ML-фильтром", "Прибыль, Просадка, ПФ + ML (многоцелевая)"],
            index=0, key="objective_choice",
            help="SQN - стабильность. Плавность Equity - ищет линейный рост. HFT Score - для HFT. Качество данных для ML - лучшая цель для WFO с ML-фильтром. Многоцелевая - компромисс. С ML-фильтром (Optuna) - обучает модель на каждой пробе, медленно."
        )
        is_multi_objective = "многоцелевая" in objective_choice

        data_files = []
        try:
            for file in os.listdir("dataCSV"):
                if file.endswith(".parquet"):
                    data_files.append(file)
        except FileNotFoundError:
            st.warning("Папка dataCSV не найдена")
        
        if data_files:
            with st.expander("Выбор данных", expanded=True):
                st.subheader("Выбор данных")
                prev_select_all_opt = st.session_state.get("select_all_data_optimization_prev", False)
                select_all = st.checkbox("Выбрать все файлы", value=prev_select_all_opt, key="select_all_data_optimization")
                
                selected_files = []
                checkbox_key_prefix = f"csv_optimization_"

                if select_all != prev_select_all_opt:
                    for file in data_files:
                        st.session_state[f"{checkbox_key_prefix}{file}"] = select_all

                for i, file in enumerate(data_files):
                    is_selected = st.checkbox(file, key=f"{checkbox_key_prefix}{file}", value=st.session_state.get(f"{checkbox_key_prefix}{file}", False))
                    if is_selected:
                        selected_files.append(file)
                
                st.session_state["select_all_data_optimization_prev"] = select_all
                st.session_state["selected_files_optimization"] = selected_files
        else:
            st.info("В папке dataCSV нет Parquet-файлов (.parquet)")
            st.session_state["selected_files_optimization"] = []
        
    dataframes = load_and_validate_data_files(selected_files, "optimization")
            
    if st.button("Запустить итеративную оптимизацию", key="run_iterative_opt", help="Запускает двухэтапную оптимизацию: сначала широкий поиск, затем углубленный с авто-корректировкой диапазонов."):
        if not dataframes:
            st.warning("Пожалуйста, выберите хотя бы один CSV-файл для оптимизации.")
        else:
            with st.spinner("Подготовка к итеративной оптимизации..."):
                combined_df = pd.concat(dataframes, ignore_index=True)
                combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
                combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

                param_space = get_param_space_from_ui()

                # Собираем все базовые (неоптимизируемые) настройки в один словарь
                base_settings = {
                    'position_size': position_size,
                    'commission': commission,
                    'aggressive_mode': st.session_state.get("aggressive_mode_optimization", False),
                    # Даты start_date и end_date не используются в самой оптимизации,
                    # поэтому их не включаем, чтобы избежать ошибки сериализации JSON.
                }

                # Выбор целевой функции в зависимости от выбора пользователя
                if "SQN" in objective_choice and not is_multi_objective:
                    strategy_objective_func = strategy_objectives.trading_strategy_objective_sqn
                elif "Плавность Equity" in objective_choice:
                    strategy_objective_func = strategy_objectives.trading_strategy_objective_equity_curve_linearity
                elif "HFT" in objective_choice:
                    strategy_objective_func = strategy_objectives.trading_strategy_objective_hft_score
                elif "Качество данных для ML" in objective_choice:
                    strategy_objective_func = strategy_objectives.trading_strategy_objective_ml_data_quality
                elif "ML" in objective_choice:
                    strategy_objective_func = strategy_objectives.trading_strategy_objective_ml
                elif "ML" in objective_choice and is_multi_objective:
                    strategy_objective_func = strategy_objectives.trading_strategy_multi_objective_ml
                else: # Многоцелевая
                    strategy_objective_func = strategy_objectives.trading_strategy_multi_objective

                iterative_params = {
                    'data': combined_df, 'param_space': param_space, 'n_trials': optuna_trials, 'base_settings': base_settings,
                    'strategy_func': strategy_objective_func,
                    'direction': 'maximize' if not is_multi_objective else ['maximize', 'maximize', 'maximize'],
                    'target_metric_value': target_metric_value
                }

                final_results = optuna_optimizer.run_iterative_optimization(iterative_params)

                stop_file = 'stop_optimization.flag'
                if os.path.exists(stop_file):
                    try:
                        os.remove(stop_file)
                    except OSError as e:
                        st.warning(f"Не удалось удалить файл-флаг остановки: {e}")

                if final_results and final_results.get('best_value') is not None:
                    if is_multi_objective:
                        st.success(f"Итеративная оптимизация завершена! Найдено {len(final_results.get('top_10_results', []))} компромиссных решений.")
                    else:
                        st.success(f"Итеративная оптимизация завершена! Лучший коэффициент стабильности: {final_results['best_value']:.4f}")
                    
                    st.subheader("Лучшие параметры (по стабильности)")
                    best_params_df = pd.DataFrame(list(final_results['best_params'].items()), columns=['Параметр', 'Значение'])
                    # Преобразуем все значения в строки, чтобы избежать ошибки Arrow с типами
                    st.dataframe(best_params_df.astype(str), use_container_width=True)
                    
                    if 'top_10_results' in final_results and final_results['top_10_results']:
                        st.subheader("Топ-10 результатов (отсортировано по стабильности)")
                        top_10_df = pd.DataFrame(final_results['top_10_results'])

                        # --- Переупорядочиваем столбцы для лучшей читаемости ---
                        desired_order = [
                            "trial_number", "value", "total_pnl", "total_trades", "win_rate", 
                            "max_drawdown", "profit_factor", "SQN"
                        ]
                        # Отбираем существующие колонки из желаемого порядка
                        available_cols = [col for col in desired_order if col in top_10_df.columns]
                        # Добавляем остальные колонки (параметры) в конец
                        remaining_cols = [col for col in top_10_df.columns if col not in available_cols]
                        top_10_df = top_10_df[available_cols + remaining_cols]
                        # --- Конец блока переупорядочивания ---

                        if is_multi_objective:
                            st.subheader("Фронт Парето")
                            pareto_df = pd.DataFrame(final_results['top_10_results'])
                            pareto_df['Max Drawdown'] = pareto_df['max_drawdown'] * 100
                            pareto_df['SQN'] = pareto_df['value'].apply(lambda x: x[0] if isinstance(x, list) else x)

                            fig = px.scatter(
                                pareto_df, x="Max Drawdown", y="SQN",
                                hover_data=pareto_df.columns,
                                title="Компромисс между стабильностью (SQN), риском (Max Drawdown) и эффективностью сигналов"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(top_10_df, use_container_width=True) # Отображаем таблицу для всех случаев

                    # --- Новые графики для анализа оптимизации ---
                    study = final_results.get('study')
                    if study:
                        st.subheader("Анализ процесса оптимизации")
                        try:
                            fig_importance = optuna.visualization.plot_param_importances(study)
                            st.plotly_chart(fig_importance, use_container_width=True)

                            fig_history = optuna.visualization.plot_optimization_history(study)
                            st.plotly_chart(fig_history, use_container_width=True)

                            # --- НОВЫЙ БЛОК: Дополнительные графики анализа параметров ---
                            st.subheader("Анализ влияния параметров")
                            optimized_params = list(study.best_params.keys())

                            if optimized_params:
                                # 1. График срезов (Slice Plot)
                                st.markdown("#### Срезовый график (Slice Plot)")
                                st.info("Этот график показывает, как меняется целевая метрика при изменении одного параметра, в то время как остальные зафиксированы на лучших значениях.")
                                slice_param = st.selectbox("Выберите параметр для срезового графика", options=optimized_params, key="slice_param_select")
                                if slice_param:
                                    fig_slice = optuna.visualization.plot_slice(study, params=[slice_param])
                                    st.plotly_chart(fig_slice, use_container_width=True)

                                # 2. Контурный график (Contour Plot)
                                if len(optimized_params) >= 2:
                                    st.markdown("#### Контурный график (Contour Plot)")
                                    st.info("Этот график показывает зависимость целевой метрики от двух параметров одновременно. Помогает найти их взаимосвязи.")
                                    cols_contour = st.columns(2)
                                    with cols_contour[0]:
                                        contour_param_x = st.selectbox("Выберите параметр для оси X", options=optimized_params, index=0, key="contour_x_select")
                                    with cols_contour[1]:
                                        contour_param_y = st.selectbox("Выберите параметр для оси Y", options=[p for p in optimized_params if p != contour_param_x], index=min(1, len(optimized_params)-2), key="contour_y_select")
                                    if contour_param_x and contour_param_y:
                                        fig_contour = optuna.visualization.plot_contour(study, params=[contour_param_x, contour_param_y])
                                        st.plotly_chart(fig_contour, use_container_width=True)
                            # --- Конец нового блока ---

                        except Exception as e:
                            st.warning(f"Не удалось построить графики анализа Optuna: {e}")

                else:
                    st.error("Итеративная оптимизация завершилась, но не удалось найти ни одного подходящего набора параметров.")

                _save_optimization_run("ITERATIVE", final_results, selected_files, base_settings, start_date, end_date)

    st.markdown("---")
    st.subheader("Walk-Forward Optimization (WFO)")
    with st.expander("Настройки WFO", expanded=True):
        wfo_unit_cols = st.columns(4)
        with wfo_unit_cols[0]:
            wfo_unit = st.selectbox("Единица измерения WFO", ["Дни", "Часы"], key="wfo_unit")
        
        unit_label = "дни" if wfo_unit == "Дни" else "часы"
        default_train = 1 if wfo_unit == "Дни" else 24 # 1 дней
        default_test = 1 if wfo_unit == "Дни" else 24 # 1 день
        default_step = 1 if wfo_unit == "Дни" else 24 # 1 день

        with wfo_unit_cols[1]:
            train_period = st.number_input(f"Окно обучения ({unit_label})", value=default_train, min_value=1, step=1, key="wfo_train_period")
        with wfo_unit_cols[2]:
            test_period = st.number_input(f"Окно теста ({unit_label})", value=default_test, min_value=1, step=1, key="wfo_test_period")
        with wfo_unit_cols[3]:
            step_period = st.number_input(f"Шаг сдвига ({unit_label})", value=default_step, min_value=1, step=1, key="wfo_step_period")

        wfo_cols = st.columns(4) # Новая строка для остальных настроек
        with wfo_cols[0]:
            trials_per_step = st.number_input("Проб на шаге", value=25, min_value=5, step=5, key="wfo_trials_per_step")
        with wfo_cols[1]:
            st.number_input("Мин. сделок для WFO", value=10, min_value=1, step=1, key="wfo_min_trades_threshold", help="Минимальное количество сделок на обучающем отрезке, чтобы результат считался значимым. Если сделок меньше, стратегия получит штраф.")

    # --- УПРОЩЕНИЕ: Оставляем одну кнопку для запуска WFO в режиме сравнения ---
    # --- НОВЫЙ БЛОК: Кнопка для автоматического подбора диапазонов и запуска WFO ---
    st.markdown("---")
    st.subheader("🚀 WFO с автоматическим подбором диапазонов")
    
    auto_wfo_cols = st.columns([2, 1, 1, 1])
    with auto_wfo_cols[0]:
        run_auto_wfo_button = st.button(
            "Запустить авто-подбор диапазонов и WFO",
            key="run_auto_wfo",
            type="primary",
            help="Запускает двухэтапный процесс: 1. Широкая оптимизация на части данных для поиска робастных диапазонов. 2. Запуск WFO с найденными диапазонами."
        )
    with auto_wfo_cols[1]:
        pre_opt_period_pct = st.slider(
            "Период для разведки (%)", 
            min_value=10, max_value=90, value=50, step=5, 
            key="pre_opt_period_pct",
            help="Процент от начала выбранных данных, который будет использован для первоначального широкого поиска диапазонов."
        )
    with auto_wfo_cols[2]:
        top_trials_pct = st.slider(
            "Топ проб для анализа (%)",
            min_value=5, max_value=50, value=10, step=5,
            key="top_trials_pct_for_ranges",
            help="Процент лучших проб из этапа разведки, которые будут использованы для формирования новых, робастных диапазонов."
        )
    with auto_wfo_cols[3]:
        pre_opt_trials = st.number_input(
            "Проб для разведки",
            value=100, min_value=10, step=10,
            key="pre_opt_trials",
            help="Количество проб Optuna для этапа широкого поиска (разведки)."
        )

    if run_auto_wfo_button:
        if not dataframes:
            st.warning("Пожалуйста, выберите хотя бы один Parquet-файл для WFO.")
        else:
            # Собираем все параметры, как для обычного WFO
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

            param_space = get_param_space_from_ui()
            base_settings = {
                'position_size': position_size, 'commission': commission,
                'aggressive_mode': st.session_state.get("aggressive_mode_optimization", False),
                'model_type': st.session_state.get("ml_model_type_optimization", "CatBoost")
            }
            wfo_params = {
                'train_period': st.session_state.get("wfo_train_period", 7),
                'test_period': st.session_state.get("wfo_test_period", 1),
                'step_period': st.session_state.get("wfo_step_period", 1),
                'trials_per_step': st.session_state.get("wfo_trials_per_step", 25),
                'wfo_unit': st.session_state.get("wfo_unit", "Дни"),
                'min_trades_threshold': st.session_state.get("wfo_min_trades_threshold", 10),
            }
            # Выбираем целевую функцию
            strategy_objective_func = _get_objective_func(objective_choice, is_multi_objective)

            # Запускаем новый процесс-оркестратор
            wfo_optimizer.run_wfo_with_auto_ranges(
                full_data=combined_df,
                wfo_params=wfo_params,
                base_settings=base_settings,
                initial_param_space=param_space,
                strategy_objective_func=strategy_objective_func,
                pre_opt_period_pct=pre_opt_period_pct,
                top_trials_pct_for_ranges=top_trials_pct,
                n_trials_pre_opt=pre_opt_trials
            )
    # --- КОНЕЦ НОВОГО БЛОКА ---

    run_wfo_comparison_button = st.button(
        "⚡ Запустить WFO (ML vs. Baseline)",
        key="run_wfo_comparison",
        help="Запускает два WFO-прогона (с ML и без) и сравнивает их результаты. Это самый быстрый и рекомендуемый режим."
    )

    if run_wfo_comparison_button:
        if not dataframes:
            st.warning("Пожалуйста, выберите хотя бы один Parquet-файл для WFO.")
        else:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

            param_space = get_param_space_from_ui()

            # Собираем базовые настройки для передачи в WFO
            base_settings = {
                'position_size': position_size, 'commission': commission,
                'aggressive_mode': st.session_state.get("aggressive_mode_optimization", False),
                'model_type': st.session_state.get("ml_model_type_optimization", "CatBoost") # <--- Добавляем тип модели
            }
            opt_params_for_wfo = {
                'param_space': param_space,
                'direction': 'maximize' if not is_multi_objective else ['maximize', 'maximize', 'maximize'],
                'base_settings': base_settings
            }

            # Собираем параметры WFO и скрининга
            wfo_params = {
                'train_period': st.session_state.get("wfo_train_period", 7),
                'test_period': st.session_state.get("wfo_test_period", 1),
                'step_period': st.session_state.get("wfo_step_period", 1),
                'trials_per_step': st.session_state.get("wfo_trials_per_step", 25),
                'wfo_unit': st.session_state.get("wfo_unit", "Дни"),
                'min_trades_threshold': st.session_state.get("wfo_min_trades_threshold", 10),
            }

            # --- Логика для режима сравнения ---
            st.header("Сравнение WFO: ML-фильтр vs. Baseline")

            # --- ИСПРАВЛЕНИЕ: Выбираем целевую функцию на основе выбора пользователя ---
            strategy_objective_func = _get_objective_func(objective_choice, is_multi_objective)

            # --- Запуск 1: WFO с ML-фильтром ---
            st.subheader(f"1. Запуск WFO с ML-фильтром (цель: '{objective_choice}')")
            opt_params_ml = opt_params_for_wfo.copy()
            opt_params_ml['strategy_func'] = strategy_objective_func
            # Добавляем флаг, чтобы wfo_optimizer знал, что нужно применить ML
            opt_params_ml['is_ml_wfo'] = True
            results_ml = wfo_optimizer.run_wfo_parallel(combined_df, wfo_params, opt_params_ml)
            if not results_ml or not results_ml['summary']:
                st.error("Прогон WFO с ML-фильтром не дал результатов. Сравнение прервано.")
                st.stop()
            st.success("Прогон с ML-фильтром завершен.")

            # --- Запуск 2: WFO без ML (Baseline) ---
            st.subheader(f"2. Запуск WFO без ML-фильтра (Baseline, цель: '{objective_choice}')")
            opt_params_no_ml = opt_params_for_wfo.copy()
            opt_params_no_ml['strategy_func'] = strategy_objective_func
            results_no_ml = wfo_optimizer.run_wfo_parallel(combined_df, wfo_params, opt_params_no_ml)
            if not results_no_ml or not results_no_ml['summary']:
                st.error("Прогон WFO без ML-фильтра не дал результатов. Сравнение прервано.")
                st.stop()
            st.success("Прогон без ML-фильтра завершен.")

            # --- Отображение результатов ---
            st.header("Итоги сравнения")

            # Сравнительная таблица метрик
            comparison_metrics_df = pd.DataFrame([
                {"Метод": "С ML-фильтром", **results_ml['aggregated_metrics']},
                {"Метод": "Без ML (Baseline)", **results_no_ml['aggregated_metrics']}
            ]).set_index("Метод")
            st.dataframe(comparison_metrics_df.T) # Транспонируем для лучшей читаемости

            # Сравнительный график Equity
            fig_comp = visualizer.plot_wfo_comparison(results_ml, results_no_ml)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Дополнительные графики для анализа ML
            summary_ml_df = pd.DataFrame(results_ml.get('summary', []))
            summary_no_ml_df = pd.DataFrame(results_no_ml.get('summary', []))

            # График эффективности ML-фильтра (сколько сделок отфильтровано)
            # и сравнение Win Rate по шагам.
            fig_ml_eff = visualizer.plot_wfo_ml_effectiveness(summary_ml_df, summary_no_ml_df)
            st.plotly_chart(fig_ml_eff, use_container_width=True)
    
            # NEW: Plot risk metrics per step
            if not summary_ml_df.empty:
                st.subheader("Метрики риска и доходности по шагам WFO (с ML-фильтром)")
                fig_risk_ml = visualizer.plot_wfo_risk_metrics(summary_ml_df)
                if fig_risk_ml:
                    st.plotly_chart(fig_risk_ml, use_container_width=True)
            
            if not summary_no_ml_df.empty:
                st.subheader("Метрики риска и доходности по шагам WFO (без ML-фильтра)")
                fig_risk_no_ml = visualizer.plot_wfo_risk_metrics(summary_no_ml_df)
                if fig_risk_no_ml:
                    st.plotly_chart(fig_risk_no_ml, use_container_width=True)
            
            # --- НОВЫЙ БЛОК: Сохранение отчета ---
            st.subheader("Сохранение отчета")
            report_name = f"WFO_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with st.spinner("Генерация HTML-отчета..."):
                html_report = visualizer.create_wfo_report_html(
                    results_ml=results_ml,
                    results_no_ml=results_no_ml,
                    param_space=param_space,
                    run_name=report_name
                )
            
            st.download_button(
                label="📥 Скачать HTML-отчет",
                data=html_report,
                file_name=f"{report_name}.html",
                mime="text/html",
                help="Сохранить все результаты и графики этого сравнения в один HTML-файл."
            )
            # --- Конец нового блока ---

def _get_objective_func(objective_choice, is_multi_objective):
    """Вспомогательная функция для выбора целевой функции по названию из UI."""
    if "SQN" in objective_choice and "ML" in objective_choice and is_multi_objective:
    if "Calmar" in objective_choice and "ML" in objective_choice:
        return strategy_objectives.trading_strategy_objective_calmar_ml
    elif "Calmar" in objective_choice:
        return strategy_objectives.trading_strategy_objective_calmar
    elif "Прибыль" in objective_choice and "ML" in objective_choice and is_multi_objective:
        return strategy_objectives.trading_strategy_multi_objective_advanced_ml
    elif "Прибыль" in objective_choice and is_multi_objective:
        return strategy_objectives.trading_strategy_multi_objective_advanced
    elif "SQN" in objective_choice and "ML" in objective_choice and is_multi_objective:
        return strategy_objectives.trading_strategy_multi_objective_ml
    elif "SQN" in objective_choice and "ML" in objective_choice:
        return strategy_objectives.trading_strategy_objective_ml
    elif "SQN" in objective_choice and is_multi_objective:
        return strategy_objectives.trading_strategy_multi_objective
    elif "SQN" in objective_choice:
        return strategy_objectives.trading_strategy_objective_sqn
    elif "Плавность Equity" in objective_choice:
        return strategy_objectives.trading_strategy_objective_equity_curve_linearity
    elif "HFT" in objective_choice:
        return strategy_objectives.trading_strategy_objective_hft_score
    elif "Качество данных для ML" in objective_choice:
        return strategy_objectives.trading_strategy_objective_ml_data_quality
    elif is_multi_objective: # Общий случай многоцелевой
        return strategy_objectives.trading_strategy_multi_objective
    else: # По умолчанию
        return strategy_objectives.trading_strategy_objective_sqn


    if st.session_state.get('optimization_running', False):
        if st.button("❌ Остановить оптимизацию", type="primary", key="stop_opt_button"):
            with open('stop_optimization.flag', 'w') as f:
                f.write('stop')
            st.warning("Сигнал остановки отправлен. Оптимизация завершится после текущей пробы.")
            st.session_state['optimization_running'] = False
            st.rerun()
    
    if st.button("Запустить оптимизацию", key="run_optuna_simple"):
        if not dataframes:
            st.warning("Пожалуйста, выберите хотя бы один Parquet-файл для оптимизации.")
        else:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

            param_space = get_param_space_from_ui()

            # Собираем все базовые (неоптимизируемые) настройки в один словарь
            base_settings = {
                'position_size': position_size,
                'commission': commission,
                'aggressive_mode': st.session_state.get("aggressive_mode_optimization", False),
                'model_type': st.session_state.get("ml_model_type_optimization", "CatBoost") # <--- Добавляем тип модели
            }

            st.subheader("Результаты оптимизации Optuna")
            
            # Выбор целевой функции в зависимости от выбора пользователя
            strategy_objective_func = _get_objective_func(objective_choice, is_multi_objective)

            opt_params_to_run = {
                'data': combined_df, 'param_space': param_space, 'n_trials': optuna_trials,
                'direction': 'maximize' if not is_multi_objective else ['maximize', 'maximize', 'maximize'],
                'base_settings': base_settings, 'data_files': selected_files,
                'strategy_func': strategy_objective_func, 'target_metric_value': target_metric_value
            }
            
            st.session_state['opt_params_to_run'] = opt_params_to_run
            st.session_state['optimization_running'] = True
            st.rerun()    

    if st.session_state.get('optimization_running', False):
        try:
            with st.spinner("Выполняется оптимизация... Пожалуйста, подождите. Прогресс отображается в консоли."):
                opt_params_to_run = st.session_state.get('opt_params_to_run')
                if not opt_params_to_run:
                    st.error("Не удалось найти параметры для запуска оптимизации. Пожалуйста, попробуйте снова.")
                    st.session_state['optimization_running'] = False
                    st.rerun()

                opt_results = optuna_optimizer.run_optimization(opt_params_to_run)

                if opt_results and opt_results.get('best_value') is not None:
                    if is_multi_objective:
                        st.success(f"Оптимизация завершена! Найдено {len(opt_results.get('top_10_results', []))} компромиссных решений.")
                    else:
                        st.success(f"Оптимизация завершена! Лучшее значение: {opt_results['best_value']:.4f}")
                    
                    st.subheader("Лучшие параметры")
                    best_params_df = pd.DataFrame(list(opt_results['best_params'].items()), columns=['Параметр', 'Значение'])
                    # Преобразуем все значения в строки, чтобы избежать ошибки Arrow с типами
                    st.dataframe(best_params_df.astype(str), use_container_width=True)
                    
                    if 'top_10_results' in opt_results and opt_results['top_10_results']:
                        st.subheader("Топ-10 результатов оптимизации")
                        top_10_df = pd.DataFrame(opt_results['top_10_results'])

                        # --- Переупорядочиваем столбцы для лучшей читаемости ---
                        desired_order = [
                            "trial_number", "value", "total_pnl", "total_trades", "win_rate", 
                            "max_drawdown", "profit_factor", "SQN"
                        ]
                        # Отбираем существующие колонки из желаемого порядка
                        available_cols = [col for col in desired_order if col in top_10_df.columns]
                        # Добавляем остальные колонки (параметры) в конец
                        remaining_cols = [col for col in top_10_df.columns if col not in available_cols]
                        top_10_df = top_10_df[available_cols + remaining_cols]
                        # --- Конец блока переупорядочивания ---

                        # Отображаем таблицу для всех случаев, когда есть top_10_results
                        st.dataframe(top_10_df, use_container_width=True)

                        if is_multi_objective:
                            st.subheader("Фронт Парето")
                            pareto_df = top_10_df.copy()
                            pareto_df['SQN'] = pareto_df['value'].apply(lambda x: x[0])
                            pareto_df['Max Drawdown'] = pareto_df['value'].apply(lambda x: -x[1] * 100)

                            fig = px.scatter(
                                pareto_df, x="Max Drawdown", y="SQN",
                                hover_data=['trial_number', 'total_pnl', 'win_rate', 'total_trades'],
                                title="Компромисс между стабильностью (SQN) и риском (Max Drawdown)"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Добавляем графики анализа
                        study = opt_results.get('study')
                        if study:
                            st.subheader("Анализ процесса оптимизации")
                            fig_importance = optuna.visualization.plot_param_importances(study)
                            st.plotly_chart(fig_importance, use_container_width=True)

                            # --- НОВЫЙ БЛОК: Дополнительные графики анализа параметров ---
                            st.subheader("Анализ влияния параметров")
                            optimized_params = list(study.best_params.keys())

                            if optimized_params:
                                # 1. График срезов (Slice Plot)
                                st.markdown("#### Срезовый график (Slice Plot)")
                                st.info("Этот график показывает, как меняется целевая метрика при изменении одного параметра, в то время как остальные зафиксированы на лучших значениях.")
                                slice_param_simple = st.selectbox("Выберите параметр для срезового графика", options=optimized_params, key="slice_param_select_simple")
                                if slice_param_simple:
                                    fig_slice_simple = optuna.visualization.plot_slice(study, params=[slice_param_simple])
                                    st.plotly_chart(fig_slice_simple, use_container_width=True)

                                # 2. Контурный график (Contour Plot)
                                if len(optimized_params) >= 2:
                                    st.markdown("#### Контурный график (Contour Plot)")
                                    st.info("Этот график показывает зависимость целевой метрики от двух параметров одновременно. Помогает найти их взаимосвязи.")
                                    cols_contour_simple = st.columns(2)
                                    with cols_contour_simple[0]:
                                        contour_param_x_simple = st.selectbox("Выберите параметр для оси X", options=optimized_params, index=0, key="contour_x_select_simple")
                                    with cols_contour_simple[1]:
                                        # Убедимся, что второй параметр не совпадает с первым
                                        available_y_params = [p for p in optimized_params if p != contour_param_x_simple]
                                        contour_param_y_simple = st.selectbox("Выберите параметр для оси Y", options=available_y_params, index=min(0, len(available_y_params)-1), key="contour_y_select_simple")
                                    if contour_param_x_simple and contour_param_y_simple:
                                        fig_contour_simple = optuna.visualization.plot_contour(study, params=[contour_param_x_simple, contour_param_y_simple])
                                        st.plotly_chart(fig_contour_simple, use_container_width=True)
                            # --- Конец нового блока ---
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if len(selected_files) == 1:
                    filename = selected_files[0]
                    dash_pos = filename.find('-')
                    data_prefix = filename[:dash_pos] if dash_pos != -1 else filename.rsplit('.', 1)[0]
                elif len(selected_files) > 1:
                    data_prefix = "ALL"
                else:
                    data_prefix = ""
                
                _save_optimization_run("SIMPLE", opt_results, selected_files, base_settings, start_date, end_date)

        except optuna.exceptions.TrialPruned as e:
            # Обрабатываем случай, когда все пробы были неудачными
            st.error(f"Оптимизация была прервана, так как не удалось получить ни одного валидного результата. Последняя ошибка: {e}")
        except Exception as e:
            st.error(f"Ошибка при запуске Optuna оптимизации: {str(e)}")
        finally:
            st.session_state['optimization_running'] = False
            if 'opt_params_to_run' in st.session_state:
                del st.session_state['opt_params_to_run']
            stop_file = 'stop_optimization.flag'
            if os.path.exists(stop_file):
                os.remove(stop_file)
            st.rerun()