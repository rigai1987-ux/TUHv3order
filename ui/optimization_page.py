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
    }

    # --- Новая логика для добавления классификаторов ---
    classifier_choices = st.session_state.get("classifier_choices", [])
    if classifier_choices:
        param_space['classifier_type'] = ('categorical', classifier_choices)

        # Определяем условные параметры для каждого классификатора
        conditional_params = {}
        if "CatBoost" in classifier_choices:
            conditional_params["CatBoost"] = {
                'catboost_iterations': ('int', st.session_state.get("catboost_iterations_min", 50), st.session_state.get("catboost_iterations_max", 300)),
                'catboost_depth': ('int', st.session_state.get("catboost_depth_min", 4), st.session_state.get("catboost_depth_max", 8)),
                'catboost_learning_rate': ('float', st.session_state.get("catboost_learning_rate_min", 0.01), st.session_state.get("catboost_learning_rate_max", 0.2)),
            }
        
        # Добавляем параметры признаков, которые используются только с ML
        if "CatBoost" in classifier_choices:
            param_space['prints_analysis_period'] = ('int', st.session_state.get("prints_analysis_period_min", 2), st.session_state.get("prints_analysis_period_max", 10))
            param_space['prints_threshold_ratio'] = ('float', st.session_state.get("prints_threshold_ratio_min", 1.1), st.session_state.get("prints_threshold_ratio_max", 3.0))
            param_space['m_analysis_period'] = ('int', st.session_state.get("m_analysis_period_min", 2), st.session_state.get("m_analysis_period_max", 10))
            param_space['m_threshold_ratio'] = ('float', st.session_state.get("m_threshold_ratio_min", 1.1), st.session_state.get("m_threshold_ratio_max", 3.0))
            param_space['hldir_window'] = ('int', st.session_state.get("hldir_window_min", 5), st.session_state.get("hldir_window_max", 20))
            param_space['hldir_offset'] = ('int', st.session_state.get("hldir_offset_min_optimization", 0), st.session_state.get("hldir_offset_max_optimization", 10))

        param_space['classifier_params'] = ('conditional', 'classifier_type', conditional_params)

    # entry_logic_mode = st.session_state.get("entry_logic_mode_optimization", "Принты и HLdir") # Логика входа теперь одна
    for name, (ptype, min_key, max_key) in param_definitions.items():
        min_val = st.session_state.get(min_key)
        max_val = st.session_state.get(max_key)
        if min_val is not None and max_val is not None:
            param_space[name] = (ptype, min_val, max_val)

    # Обработка логики входа
    param_space["entry_logic_mode"] = ("categorical", ["Вилка отложенных ордеров"])

    # Обработка параметров "вилки"
    param_space["bracket_offset_pct"] = ("float", st.session_state.get("bracket_offset_pct_min_optimization"), st.session_state.get("bracket_offset_pct_max_optimization"))
    param_space["bracket_timeout_candles"] = ("int", st.session_state.get("bracket_timeout_candles_min_optimization"), st.session_state.get("bracket_timeout_candles_max_optimization"))

    # Обработка климаксного выхода
    use_climax_exit_option = st.session_state.get("use_climax_exit_option", "Нет")
    if use_climax_exit_option == "Да":
        param_space["use_climax_exit"] = ("categorical", [True])
        param_space["climax_exit_window"] = ("int", st.session_state.get("climax_exit_window_min_optimization"), st.session_state.get("climax_exit_window_max_optimization"))
        param_space["climax_exit_threshold"] = ("float", st.session_state.get("climax_exit_threshold_min_optimization"), st.session_state.get("climax_exit_threshold_max_optimization"))
    else:
        param_space["use_climax_exit"] = ("categorical", [False], None)

    # Удаляем None значения, если виджеты не были созданы
    return {k: v for k, v in param_space.items() if all(i is not None for i in v[1:])}


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

        manage_profiles("optimization", get_optimization_parameters)
    
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
                value=float(st.session_state.get("bracket_offset_pct_min_optimization", 0.1)),
                min_value=0.01, step=0.01, key="bracket_offset_pct_min_optimization", format="%.2f"
            )
        with min_cols[1]:
            st.number_input(
                "Тайм-аут (min, свечи)",
                value=int(st.session_state.get("bracket_timeout_candles_min_optimization", 2)),
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

        st.markdown("**🚀 Климаксный выход**")
        use_climax_exit_option = st.radio("Оптимизировать климаксный выход?", ("Да", "Нет"), index=1, key="use_climax_exit_option")
        if use_climax_exit_option == "Да":
            min_cols, max_cols = st.columns(2), st.columns(2)
            with min_cols[0]:
                st.number_input("Окно (min)", value=int(st.session_state.get("climax_exit_window_min_optimization", 5)), min_value=5, step=1, key="climax_exit_window_min_optimization")
            with min_cols[1]:
                st.number_input("Порог (min)", value=float(st.session_state.get("climax_exit_threshold_min_optimization", 1.0)), min_value=0.1, step=0.1, key="climax_exit_threshold_min_optimization", format="%.1f")
            with max_cols[0]:
                st.number_input("Окно (max)", value=int(st.session_state.get("climax_exit_window_max_optimization", 100)), min_value=5, step=1, key="climax_exit_window_max_optimization")
            with max_cols[1]:
                st.number_input("Порог (max)", value=float(st.session_state.get("climax_exit_threshold_max_optimization", 15.0)), min_value=0.1, step=0.1, key="climax_exit_threshold_max_optimization", format="%.1f")
        
        # --- Новый блок для выбора и настройки классификаторов ---
        st.markdown("---")
        st.markdown("### 🤖 Оптимизация классификаторов")
        
        # Используем toggle, чтобы можно было полностью отключить ML-часть
        use_ml_classifiers = st.toggle("Использовать ML классификаторы", value=False, key="use_ml_classifiers", help="Если включено, Optuna будет также выбирать лучший классификатор и его параметры.")

        if use_ml_classifiers:
            classifier_choices = st.multiselect(
                "Выберите классификатор для оптимизации",
                options=["CatBoost"],
                default=["CatBoost"] if st.session_state.get("use_ml_classifiers") else [],
                key="classifier_choices"
            )

            if "CatBoost" in classifier_choices:
                with st.expander("Настройки CatBoost"):
                    cb_cols1, cb_cols2 = st.columns(3), st.columns(3)
                    with cb_cols1[0]: st.number_input("iterations (min)", 50, key="catboost_iterations_min")
                    with cb_cols1[1]: st.number_input("depth (min)", 4, key="catboost_depth_min")
                    with cb_cols1[2]: st.number_input("learning_rate (min)", 0.01, format="%.3f", step=0.001, key="catboost_learning_rate_min")
                    with cb_cols2[0]: st.number_input("iterations (max)", 300, key="catboost_iterations_max")
                    with cb_cols2[1]: st.number_input("depth (max)", 8, key="catboost_depth_max")
                    with cb_cols2[2]: st.number_input("learning_rate (max)", 0.2, format="%.3f", step=0.001, key="catboost_learning_rate_max")
            
            # Новый блок для настройки параметров признаков
            if classifier_choices:
                st.markdown("##### Настройки признаков для ML")
                st.info("Эти параметры определяют, как будут рассчитываться индикаторы, используемые моделью в качестве признаков.")
                feat_cols1, feat_cols2 = st.columns(2), st.columns(2)
                with feat_cols1[0]: st.number_input("Prints Period (min/max)", 2, key="prints_analysis_period_min", help="Период для анализа соотношения принтов.")
                with feat_cols2[0]: st.number_input("Max Prints Period", 10, key="prints_analysis_period_max", label_visibility="collapsed")
                with feat_cols1[1]: st.number_input("Prints Ratio (min/max)", 1.1, step=0.1, format="%.2f", key="prints_threshold_ratio_min", help="Порог соотношения long/short принтов.")
                with feat_cols2[1]: st.number_input("Max Prints Ratio", 3.0, step=0.1, format="%.2f", key="prints_threshold_ratio_max", label_visibility="collapsed")
                
                feat_cols3, feat_cols4 = st.columns(2), st.columns(2)
                with feat_cols3[0]: st.number_input("M-Ratio Period (min/max)", 2, key="m_analysis_period_min", help="Период для анализа M-Ratio.")
                with feat_cols4[0]: st.number_input("Max M-Ratio Period", 10, key="m_analysis_period_max", label_visibility="collapsed")
                with feat_cols3[1]: st.number_input("M-Ratio (min/max)", 1.1, step=0.1, format="%.2f", key="m_threshold_ratio_min", help="Порог соотношения long/short M-Ratio.")
                with feat_cols4[1]: st.number_input("Max M-Ratio", 3.0, step=0.1, format="%.2f", key="m_threshold_ratio_max", label_visibility="collapsed")

                feat_cols5, feat_cols6 = st.columns(2), st.columns(2) # Новая строка для HLdir
                with feat_cols5[0]: st.number_input("HLdir Window (min/max)", 5, key="hldir_window_min_optimization", help="Окно сглаживания для индикатора HLdir.")
                with feat_cols6[0]: st.number_input("Max HLdir Window", 20, key="hldir_window_max_optimization", label_visibility="collapsed")
                with feat_cols5[1]: st.number_input("HLdir Offset (min/max)", 0, key="hldir_offset_min_optimization", help="Смещение (shift) для индикатора HLdir. Положительное значение - сдвиг в прошлое.")
                with feat_cols6[1]: st.number_input("Max HLdir Offset", 10, key="hldir_offset_max_optimization", label_visibility="collapsed")


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

        backend_choice = st.selectbox(
            "Бэкенд для параллелизма",
            options=["threading", "loky"], index=0, key="backend_choice",
            help="`threading` - быстрее для коротких задач (рекомендуется). `loky` - надежнее, использует процессы вместо потоков (как в WFO Турбо)."
        )

        objective_choice = st.selectbox(
            "Цель оптимизации",
            options=["SQN (стабильность)", "HFT Score (частота и стабильность)", "SQN, Max Drawdown и Эффективность сигналов (многоцелевая)"],
            index=2, key="objective_choice",
            help="SQN - ищет стабильные стратегии. HFT Score - для высокочастотных стратегий. Многоцелевая - ищет компромисс между стабильностью, риском и эффективностью."
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
                elif "HFT" in objective_choice:
                    strategy_objective_func = strategy_objectives.trading_strategy_objective_hft_score
                else: # Многоцелевая
                    strategy_objective_func = strategy_objectives.trading_strategy_multi_objective

                iterative_params = {
                    'data': combined_df, 'param_space': param_space, 'n_trials': optuna_trials, 'base_settings': base_settings,
                    'strategy_func': strategy_objective_func,
                    'direction': 'maximize' if not is_multi_objective else ['maximize', 'maximize', 'maximize'],
                    'target_metric_value': target_metric_value,
                    'backend_choice': backend_choice
                }

                # --- Автоматический выбор бэкенда при использовании ML ---
                use_ml = bool(st.session_state.get("classifier_choices"))
                if use_ml and backend_choice != 'loky':
                    st.warning(
                        "Для оптимизации с ML-классификаторами автоматически выбран бэкенд 'loky' для обеспечения "
                        "настоящего параллелизма при обучении моделей. Ваш выбор 'threading' был проигнорирован."
                    )
                    iterative_params['backend_choice'] = 'loky'

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
                            st.plotly_chart(fig, width='stretch')
                        st.dataframe(top_10_df, width='stretch') # Отображаем таблицу для всех случаев

                    # --- Новые графики для анализа оптимизации ---
                    study = final_results.get('study')
                    if study:
                        st.subheader("Анализ процесса оптимизации")
                        try:
                            fig_importance = optuna.visualization.plot_param_importances(study)
                            st.plotly_chart(fig_importance, width='stretch')

                            fig_history = optuna.visualization.plot_optimization_history(study)
                            st.plotly_chart(fig_history, width='stretch')
                        except Exception as e:
                            st.warning(f"Не удалось построить графики анализа Optuna: {e}")

                else:
                    st.error("Итеративная оптимизация завершилась, но не удалось найти ни одного подходящего набора параметров.")

                # --- Начало блока сохранения результатов итеративной оптимизации ---
                if final_results and final_results.get('best_params'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    if len(selected_files) == 1:
                        filename = selected_files[0]
                        dash_pos = filename.find('-')
                        data_prefix = filename[:dash_pos] if dash_pos != -1 else filename.rsplit('.', 1)[0]
                    elif len(selected_files) > 1:
                        data_prefix = "ALL"
                    else:
                        data_prefix = ""

                    def extract_numeric_value(value_str):
                        if value_str is None or (isinstance(value_str, float) and np.isnan(value_str)):
                            return 0
                        if isinstance(value_str, (list, tuple)): # Для многоцелевой
                            value_str = value_str[0]
                        if hasattr(value_str, 'dtype') and np.issubdtype(value_str.dtype, np.number):
                            value_str = value_str.item()
                        numeric_str = ''.join(filter(lambda x: x.isdigit() or x == '.', str(value_str).replace('$', '').replace('%', '').replace('-', '')))
                        try:
                            return int(float(numeric_str) + 0.5)
                        except (ValueError, TypeError):
                            return 0

                    best_value = extract_numeric_value(final_results.get('best_value'))
                    mode_suffix = "_ITERATIVE"
                    new_run_name = f"run_{timestamp}_{data_prefix}{mode_suffix}_${best_value}_OPTUNA"
                    
                    ranges_dict = {k: v for k, v in st.session_state.items() if k.endswith('_optimization')}
                    
                    # Собираем полные базовые настройки, включая даты
                    full_base_settings = {
                        **base_settings,
                        "start_date": str(start_date), "end_date": str(end_date)
                    }
                    run_data = {
                        "run_name": new_run_name, "timestamp": datetime.now().isoformat(),
                        "ranges": ranges_dict, "settings": full_base_settings,
                        "data_files": selected_files, "best_params": final_results.get('best_params', {}),
                        "optimization_type": "optuna_iterative",
                        "top_10_results": final_results.get('top_10_results', [])
                    }
                    
                    def convert_numpy_types(obj):
                        if isinstance(obj, (datetime, date)): return obj.isoformat()
                        if isinstance(obj, np.integer): return int(obj)
                        elif isinstance(obj, np.floating): return float(obj)
                        elif isinstance(obj, np.ndarray): return obj.tolist()
                        elif isinstance(obj, dict): return {key: convert_numpy_types(value) for key, value in obj.items()}
                        elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
                        else: return obj
                    
                    run_data_converted = convert_numpy_types(run_data)
                    
                    try:
                        os.makedirs("optimization_runs", exist_ok=True)
                        file_path = f"optimization_runs/{new_run_name}.json"
                        json_data = json.dumps(run_data_converted, ensure_ascii=False, indent=2)
                        _atomic_write(file_path, json_data)
                        st.success(f"Результаты итеративной оптимизации сохранены как '{new_run_name}'")
                    except (IOError, OSError) as e:
                        st.error(f"Ошибка ввода-вывода при сохранении результатов: {str(e)}")
                    except Exception as e:
                        st.error(f"Ошибка при сохранении результатов итеративной оптимизации: {str(e)}")
                # --- Конец блока сохранения ---

    st.markdown("---")
    st.subheader("Walk-Forward Optimization (WFO)")
    with st.expander("Настройки WFO и скрининга", expanded=True):
        wfo_unit_cols = st.columns(4)
        with wfo_unit_cols[0]:
            wfo_unit = st.selectbox("Единица измерения WFO", ["Дни", "Часы"], key="wfo_unit")
        
        unit_label = "дни" if wfo_unit == "Дни" else "часы"
        default_train = 7 if wfo_unit == "Дни" else 168 # 7 дней
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

        st.markdown("##### Настройки предварительного скрининга для WFO")
        use_wfo_anchoring = st.toggle(
            "Анкерная оптимизация WFO",
            value=True,
            key="use_wfo_anchoring",
            help="Если включено, лучшие параметры с предыдущего шага WFO будут использованы как первая проба для следующего шага. Это может ускорить сходимость."
        )

        screening_cols = st.columns(3)
        with screening_cols[0]:
            look_forward_period = st.number_input("Горизонт скрининга (свечи)", value=20, min_value=1, step=1, key="wfo_look_forward_period", help="На сколько свечей вперед смотреть для определения 'перспективности' сигнала.")
        with screening_cols[1]:
            profit_target_pct = st.number_input("Цель по прибыли (%)", value=2.0, min_value=0.1, step=0.1, key="wfo_profit_target_pct", format="%.1f", help="Какой процент прибыли должен быть достигнут, чтобы сигнал считался перспективным.")
        with screening_cols[2]:
            loss_limit_pct = st.number_input("Ограничение убытка (%)", value=1.0, min_value=0.1, step=0.1, key="wfo_loss_limit_pct", format="%.1f", help="При достижении этого убытка сигнал считается неперспективным.")

        use_screening_for_wfo = st.toggle(
            "Использовать скрининг в WFO", 
            value=True, 
            key="use_screening_for_wfo",
            help="Если включено, Optuna будет обучаться только на 'перспективных' сигналах, найденных на обучающем отрезке. Это может улучшить качество оптимизации."
        )

    wfo_button_cols = st.columns([2, 2, 1])
    with wfo_button_cols[0]:
        run_wfo_button = st.button("🚀 Запустить WFO (пошагово)", key="run_wfo", help="Запускает пошаговую оптимизацию с выводом прогресса в реальном времени. Медленнее, но информативнее.")
    with wfo_button_cols[1]:
        run_wfo_parallel_button = st.button("⚡ Запустить WFO (Турбо)", key="run_wfo_parallel", help="Запускает все оптимизации WFO параллельно. Максимально быстро, но результат отображается только в конце.")
    with wfo_button_cols[2]:
        show_wfo_progress = st.toggle("Показывать прогресс", value=True, key="show_wfo_progress", help="Отключите для ускорения WFO. Прогресс будет виден в консоли, а в UI отобразится только финальный результат.")

    # Определяем, какая кнопка была нажата
    run_mode = None
    if run_wfo_button: run_mode = 'sequential'
    if run_wfo_parallel_button: run_mode = 'parallel'

    if run_mode:
        if not dataframes:
            st.warning("Пожалуйста, выберите хотя бы один Parquet-файл для WFO.")
        else:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

            param_space = get_param_space_from_ui()

            # Выбор целевой функции в зависимости от выбора пользователя
            if "SQN" in objective_choice and not is_multi_objective:
                strategy_objective_func = strategy_objectives.trading_strategy_objective_sqn
            elif "HFT" in objective_choice:
                strategy_objective_func = strategy_objectives.trading_strategy_objective_hft_score
            else: # Многоцелевая
                strategy_objective_func = strategy_objectives.trading_strategy_multi_objective

            opt_params_for_wfo = {
                'param_space': param_space,
                'direction': 'maximize' if not is_multi_objective else ['maximize', 'maximize', 'maximize'],
                'position_size': position_size, 'commission': commission,
                'aggressive_mode': st.session_state.get("aggressive_mode_optimization", False),
                'strategy_func': strategy_objective_func
            }

            # Собираем параметры WFO и скрининга
            wfo_params = {
                'train_period': st.session_state.get("wfo_train_period", 7),
                'test_period': st.session_state.get("wfo_test_period", 1),
                'step_period': st.session_state.get("wfo_step_period", 1),
                'trials_per_step': st.session_state.get("wfo_trials_per_step", 25),
                'wfo_unit': st.session_state.get("wfo_unit", "Дни"),
                'look_forward_period': st.session_state.get("wfo_look_forward_period", 20),
                'min_trades_threshold': st.session_state.get("wfo_min_trades_threshold", 10),
                'profit_target_pct': st.session_state.get("wfo_profit_target_pct", 2.0),
                'loss_limit_pct': st.session_state.get("wfo_loss_limit_pct", 1.0),
                'use_anchoring': use_wfo_anchoring, # Передаем новый параметр
            }

            if run_mode == 'sequential':
                wfo_results = wfo_optimizer.run_wfo(
                    combined_df, 
                    wfo_params, 
                    opt_params_for_wfo, 
                    use_screening=use_screening_for_wfo,
                    objective_name=objective_choice,
                    show_progress_ui=show_wfo_progress
                )
            else: # parallel mode
                wfo_results = wfo_optimizer.run_wfo_parallel(
                    combined_df, wfo_params, opt_params_for_wfo, use_screening=use_screening_for_wfo
                )

            if wfo_results and wfo_results['summary']:
                st.balloons()
                summary_df = pd.DataFrame(wfo_results['summary'])
                st.subheader("Результаты WFO")
                st.dataframe(summary_df, width='stretch')
                st.subheader("Итоговые метрики (Out-of-Sample)")
                st.json(wfo_results['aggregated_metrics'])

                fig = visualizer.plot_wfo_results(summary_df, wfo_results['equity_curve'], wfo_results['aggregated_metrics'])
                st.plotly_chart(fig, width='stretch')

                # --- Дополнительные графики для анализа WFO ---
                st.subheader("Углубленный анализ WFO")

                # 1. График стабильности параметров
                param_space = get_param_space_from_ui()
                fig_params = visualizer.plot_wfo_parameter_stability(summary_df, param_space)
                if fig_params:
                    st.plotly_chart(fig_params, width='stretch')

                # 2. График сравнения In-Sample vs Out-of-Sample
                fig_is_oos = visualizer.plot_wfo_insample_vs_outsample(summary_df)
                if fig_is_oos:
                    st.plotly_chart(fig_is_oos, width='stretch')

                # 3. График важности признаков
                fig_feat_imp = visualizer.plot_wfo_feature_importance(summary_df)
                if fig_feat_imp:
                    st.plotly_chart(fig_feat_imp, width='stretch')

                # --- Конец блока дополнительных графиков ---

                # --- Отображение и применение предложенных диапазонов ---
                if wfo_results.get('suggested_ranges'):
                    st.subheader("💡 Предлагаемые диапазоны для следующей оптимизации")
                    st.info("Эти диапазоны основаны на параметрах, показавших лучший результат на успешных (прибыльных) шагах WFO.")
                    
                    suggested_df = pd.DataFrame.from_dict(wfo_results['suggested_ranges'], orient='index')
                    suggested_df.reset_index(inplace=True)
                    suggested_df.columns = ['Параметр', 'Рекомендуемый min', 'Рекомендуемый max']
                    st.dataframe(suggested_df, width='stretch')

                    if st.button("✅ Применить предложенные диапазоны", key="apply_wfo_ranges"):
                        for _, row in suggested_df.iterrows():
                            param_name = row['Параметр']
                            min_key = f"{param_name}_min_optimization"
                            max_key = f"{param_name}_max_optimization"
                            
                            # Обновляем session_state, чтобы виджеты в сайдбаре изменились
                            if min_key in st.session_state:
                                st.session_state[min_key] = row['Рекомендуемый min']
                            if max_key in st.session_state:
                                st.session_state[max_key] = row['Рекомендуемый max']
                        st.success("Диапазоны в сайдбаре обновлены! Можете запустить новую оптимизацию.")
                        st.rerun() # Перезапускаем скрипт, чтобы виджеты обновились
                        return # Важно! Прерываем выполнение, чтобы избежать перезаписи session_state
            else:
                st.warning("WFO не дал результатов для визуализации.")
    
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
            }

            st.subheader("Результаты оптимизации Optuna")
            
            # Выбор целевой функции в зависимости от выбора пользователя
            if "SQN" in objective_choice and not is_multi_objective:
                strategy_objective_func = strategy_objectives.trading_strategy_objective_sqn
            elif "HFT" in objective_choice:
                strategy_objective_func = strategy_objectives.trading_strategy_objective_hft_score
            else: # Многоцелевая
                strategy_objective_func = strategy_objectives.trading_strategy_multi_objective

            opt_params_to_run = {
                'data': combined_df, 'param_space': param_space, 'n_trials': optuna_trials,
                'direction': 'maximize' if not is_multi_objective else ['maximize', 'maximize', 'maximize'],
                'base_settings': base_settings, 'data_files': selected_files,
                'strategy_func': strategy_objective_func,
                'target_metric_value': target_metric_value,
                'backend_choice': backend_choice
            }
            
            # --- Автоматический выбор бэкенда при использовании ML ---
            use_ml = bool(st.session_state.get("classifier_choices"))
            if use_ml and backend_choice != 'loky':
                st.warning(
                    "Для оптимизации с ML-классификаторами автоматически выбран бэкенд 'loky' для обеспечения "
                    "настоящего параллелизма при обучении моделей. Ваш выбор 'threading' был проигнорирован."
                )
                opt_params_to_run['backend_choice'] = 'loky'

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
                            st.plotly_chart(fig, width='stretch')
                        # Отображаем таблицу для всех случаев, когда есть top_10_results, а не только для многоцелевой
                        
                        # Улучшаем читаемость таблицы с параметрами классификаторов
                        if any('classifier_type' in res for res in opt_results['top_10_results']):
                            cleaned_results = [optuna_optimizer._flatten_conditional_params(res.copy()) for res in opt_results['top_10_results']]
                            top_10_df = pd.DataFrame(cleaned_results)

                        st.dataframe(top_10_df, width='stretch')

                        # Добавляем графики анализа
                        study = opt_results.get('study')
                        if study:
                            st.subheader("Анализ процесса оптимизации")
                            fig_importance = optuna.visualization.plot_param_importances(study)
                            st.plotly_chart(fig_importance, width='stretch')
                        st.dataframe(top_10_df, width='stretch')
                else:
                    st.error("Оптимизация завершилась, но не удалось найти ни одного подходящего набора параметров. Попробуйте расширить диапазоны оптимизации или проверить данные.")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if len(selected_files) == 1:
                    filename = selected_files[0]
                    dash_pos = filename.find('-')
                    data_prefix = filename[:dash_pos] if dash_pos != -1 else filename.rsplit('.', 1)[0]
                elif len(selected_files) > 1:
                    data_prefix = "ALL"
                else:
                    data_prefix = ""
                
                def extract_numeric_value(value_str):
                    if value_str is None or (isinstance(value_str, float) and np.isnan(value_str)):
                        return 0
                    if isinstance(value_str, (list, tuple)): # Для многоцелевой
                        value_str = value_str[0]
                    if hasattr(value_str, 'dtype') and np.issubdtype(value_str.dtype, np.number):
                        value_str = value_str.item()
                    numeric_str = ''.join(filter(lambda x: x.isdigit() or x == '.', str(value_str).replace('$', '').replace('%', '').replace('-', '')))
                    try:
                        return int(float(numeric_str) + 0.5)
                    except ValueError:
                        return 0
                
                best_value = extract_numeric_value(opt_results.get('best_value'))
                mode_suffix = "_mode1"
                new_run_name = f"run_{timestamp}_{data_prefix}{mode_suffix}_${best_value}_OPTUNA"
                
                ranges_dict = {k: v for k, v in st.session_state.items() if k.endswith('_optimization')}
                
                # Обновляем `settings`, чтобы включить все базовые параметры
                full_settings = {
                    **base_settings,
                    "start_date": str(start_date), "end_date": str(end_date)
                }
                optuna_results = []
                if opt_results and opt_results.get('best_params'):
                    # Используем top_10_results, которые уже содержат все нужные метрики
                    optuna_results = opt_results.get('top_10_results', [])
                
                run_data = {
                    "run_name": new_run_name, "timestamp": datetime.now().isoformat(),
                    "ranges": ranges_dict,
                    "settings": full_settings,
                    "data_files": selected_files,
                    "best_params": opt_results.get('best_params', {}),
                    "optimization_type": "optuna",
                    "top_10_results": opt_results.get('top_10_results', [])
                }
                
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, dict): return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
                    else: return obj
                
                run_data_converted = convert_numpy_types(run_data)
                
                try:
                    os.makedirs("optimization_runs", exist_ok=True)
                    file_path = f"optimization_runs/{new_run_name}.json"
                    json_data = json.dumps(run_data_converted, ensure_ascii=False, indent=2)
                    _atomic_write(file_path, json_data)
                    st.success(f"Результаты Optuna оптимизации сохранены как '{new_run_name}'")
                except (IOError, OSError) as e:
                    st.error(f"Ошибка ввода-вывода при сохранении результатов: {str(e)}")
                except Exception as e:
                    st.error(f"Ошибка при сохранении результатов Optuna оптимизации: {str(e)}")
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