import streamlit as st
import pandas as pd
from datetime import datetime
import os

import logging
# Импортируем необходимые функции из других модулей
from ml_model_handler import train_ml_model
from signal_generator import generate_signals
from trading_simulator import run_trading_simulation
from visualizer import save_figure_to_html, plot_single_trade, plot_ml_decision_boundary
from app_utils import get_basic_settings, manage_profiles, get_strategy_parameters, load_and_validate_data_files # type: ignore

def run_forced_ml_analysis(df, params):
    """
    Выполняет принудительный ML-анализ и возвращает фигуру для отображения.
    Эта функция инкапсулирует логику, которая раньше была внутри `show_analysis_page`.
    """
    logging.info("Запущен принудительный ML-анализ.")
    
    # 1. Генерируем базовые сигналы
    base_signal_indices, _, _ = generate_signals(df, params, base_signal_only=True)
    logging.info(f"Шаг 1: Найдено {len(base_signal_indices)} базовых сигналов для анализа.")

    # 2. Обучаем модель
    ml_results = train_ml_model(df, params, base_signal_indices)
    model = ml_results.get('model')
    logging.info(f"Шаг 2: Обучение модели завершено. Модель создана: {model is not None}.")

    if not model:
        reason = ml_results.get('failure_reason', "Проверьте параметры и данные.")
        logging.error(f"Шаг 2 ОШИБКА: Не удалось обучить модель. Причина: {reason}")
        st.warning(f"Не удалось обучить ML-модель. Причина: {reason}")
        return None

    # 3. Готовим данные для предсказания
    features_for_prediction = ml_results['feature_df'].loc[base_signal_indices]
    X_scaled = ml_results['scaler'].transform(features_for_prediction)
    predictions = model.predict(X_scaled)
    
    # 4. Определяем отклоненные сигналы
    rejected_mask = predictions != 1
    rejected_indices = features_for_prediction.index[rejected_mask].tolist()
    logging.info(f"Шаг 5: Найдено {len(rejected_indices)} отклоненных сигналов.")

    # 5. Собираем результаты для отрисовки
    temp_ml_results_for_plot = {'feature_importances': ml_results.get('feature_importances'), 'ml_features_df': ml_results['feature_df'], 'ml_rejected_signals': rejected_indices}
    return plot_ml_decision_boundary(df, temp_ml_results_for_plot, params)

def show_analysis_page():
    """
    Отображает страницу "Анализ сигналов".
    """
    # Базовые настройки в боковой панели
    with st.sidebar:
        st.header("Настройки анализа")
        position_size, commission, start_date, end_date = get_basic_settings("analysis")

        manage_profiles("analysis", get_strategy_parameters)
        
        st.subheader("Параметры стратегии (Вилка отложенных ордеров)")

        st.markdown("**📊 Фильтр объёма**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("vol_pctl (%)", value=float(st.session_state.get("vol_pctl_analysis", 1.0)), min_value=0.01, step=0.01, key="vol_pctl_analysis")
        with col2:
            st.number_input("vol_period", value=int(st.session_state.get("vol_period_analysis", 20)), min_value=1, step=1, key="vol_period_analysis")

        st.markdown("**📏 Фильтр диапазона**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("rng_pctl (%)", value=float(st.session_state.get("rng_pctl_analysis", 1.0)), min_value=0.01, step=0.01, key="rng_pctl_analysis")
        with col2:
            st.number_input("range_period", value=int(st.session_state.get("range_period_analysis", 20)), min_value=1, step=1, key="range_period_analysis")

        st.markdown("**📉 Фильтр NATR**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("natr_min (%)", value=float(st.session_state.get("natr_min_analysis", 0.35)), min_value=0.01, step=0.01, key="natr_min_analysis")
        with col2:
            st.number_input("natr_period", value=int(st.session_state.get("natr_period_analysis", 10)), min_value=1, step=1, key="natr_period_analysis")

        st.markdown("**📈 Фильтр роста**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("min_growth_pct (%)", value=st.session_state.get("min_growth_pct_analysis", 1.0), min_value=-100.0, max_value=100.0, step=0.01, key="min_growth_pct_analysis")
        with col2:
            st.number_input("lookback_period", value=int(st.session_state.get("lookback_period_analysis", 20)), min_value=1, step=1, key="lookback_period_analysis")

        st.markdown("**🔧 Дополнительные параметры**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("stop_loss_pct (%)", value=float(st.session_state.get("stop_loss_pct_analysis", 2.0)), min_value=0.01, step=0.01, format="%.2f", key="stop_loss_pct_analysis")
        with col2:
            st.number_input("take_profit_pct (%)", value=float(st.session_state.get("take_profit_pct_analysis", 4.0)), min_value=0.01, step=0.01, format="%.2f", key="take_profit_pct_analysis")

        st.checkbox(
            "Инвертировать направление сделок", 
            value=st.session_state.get("invert_direction_analysis", False), 
            key="invert_direction_analysis", 
            help="Если включено, все сигналы 'long' будут исполнены как 'short', и наоборот."
        )

        st.markdown("**Параметры вилки**")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Отступ вилки (%)", value=float(st.session_state.get("bracket_offset_pct_analysis", 0.5)),
                min_value=0.01, step=0.01, key="bracket_offset_pct_analysis", format="%.2f"
            )
        with col2:
            st.number_input(
                "Тайм-аут ожидания (свечи)", value=int(st.session_state.get("bracket_timeout_candles_analysis", 5)),
                min_value=1, step=1, key="bracket_timeout_candles_analysis"
            )

        st.checkbox("Использовать климаксный выход", value=st.session_state.get("use_climax_exit_analysis", False), key="use_climax_exit_analysis")
        if st.session_state.get("use_climax_exit_analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Окно", value=int(st.session_state.get("climax_exit_window_analysis", 50)), min_value=5, step=1, key="climax_exit_window_analysis")
            with col2:
                st.number_input("Порог", value=float(st.session_state.get("climax_exit_threshold_analysis", 3.0)), min_value=0.1, step=0.1, key="climax_exit_threshold_analysis")

        # --- Блок для отображения и хранения параметров ML ---
        # Этот блок будет виден, если параметры ML были переданы (например, из аналитики)
        # и будет хранить их в session_state для передачи в симулятор.
        if st.session_state.get("classifier_type_analysis"):
            st.markdown("---")
            st.subheader("Параметры ML-модели")
            st.info("Эти параметры были загружены из результатов оптимизации и будут использованы в симуляции.")

            # Отображаем и храним тип классификатора
            st.text_input("Тип классификатора", value=st.session_state.get("classifier_type_analysis"), key="classifier_type_analysis", disabled=True)

            # Отображаем и храним параметры CatBoost, если он выбран
            if st.session_state.get("classifier_type_analysis") == "CatBoost":
                st.number_input("iterations", value=int(st.session_state.get("catboost_iterations_analysis", 100)), key="catboost_iterations_analysis", disabled=True)
                st.number_input("depth", value=int(st.session_state.get("catboost_depth_analysis", 4)), key="catboost_depth_analysis", disabled=True)
                st.number_input("learning_rate", value=float(st.session_state.get("catboost_learning_rate_analysis", 0.1)), key="catboost_learning_rate_analysis", disabled=True, format="%.4f")

        # --- Новый блок для параметров признаков ML ---
        # Этот блок будет виден, если параметры ML были переданы (например, из аналитики).
        # Виджеты создаются для хранения значений в session_state, но они отключены (disabled=True).
        if st.session_state.get("classifier_type_analysis"):
            with st.expander("Параметры признаков для ML", expanded=False):
                st.info("Эти параметры используются для расчета индикаторов, которые служат признаками для ML-модели.")

                st.markdown("**Признаки на основе принтов**")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Период анализа принтов", value=int(st.session_state.get("prints_analysis_period_analysis", 5)), min_value=1, step=1, key="prints_analysis_period_analysis", disabled=True)
                with col2:
                    st.number_input("Порог соотношения принтов", value=float(st.session_state.get("prints_threshold_ratio_analysis", 1.5)), min_value=1.0, step=0.1, format="%.2f", key="prints_threshold_ratio_analysis", disabled=True)

                st.markdown("**Признаки на основе M-Ratio**")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Период анализа M-Ratio", value=int(st.session_state.get("m_analysis_period_analysis", 5)), min_value=1, step=1, key="m_analysis_period_analysis", disabled=True)
                with col2:
                    st.number_input("Порог M-Ratio", value=float(st.session_state.get("m_threshold_ratio_analysis", 1.5)), min_value=1.0, step=0.1, format="%.2f", key="m_threshold_ratio_analysis", disabled=True)

                st.markdown("**Признаки на основе HLdir**")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Окно сглаживания HLdir", value=int(st.session_state.get("hldir_window_analysis", 10)), min_value=1, step=1, key="hldir_window_analysis", disabled=True)
                with col2:
                    st.number_input("Смещение HLdir", value=int(st.session_state.get("hldir_offset_analysis", 0)), min_value=0, step=1, key="hldir_offset_analysis", disabled=True)

        # --- Конец параметров стратегии ---

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
                prev_select_all = st.session_state.get("select_all_data_prev", False)
                select_all = st.checkbox("Выбрать все файлы", value=prev_select_all, key="select_all_data")
                
                selected_files = []
                checkbox_key_prefix = f"csv_analysis_"
                
                if select_all != prev_select_all:
                    for file in data_files:
                        st.session_state[f"{checkbox_key_prefix}{file}"] = select_all
                
                for i, file in enumerate(data_files):
                    is_selected = st.checkbox(file, key=f"{checkbox_key_prefix}{file}", value=st.session_state.get(f"{checkbox_key_prefix}{file}", False))
                    if is_selected:
                        selected_files.append(file)
                
                st.session_state["select_all_data_prev"] = select_all
                st.session_state["selected_files_analysis"] = selected_files
        else:
            st.info("В папке dataCSV нет Parquet-файлов (.parquet)")
            st.session_state["selected_files_analysis"] = []
    
    selected_files = st.session_state.get("selected_files_analysis", [])
    if selected_files:
        st.write(f"Выбраны файлы: {', '.join(selected_files)}")
    else:
        st.write("Файлы не выбраны")
    
    dataframes = load_and_validate_data_files(selected_files, "analysis")

    # --- НОВЫЙ БЛОК: Автоматический запуск анализа при переходе из "Аналитики" ---
    if st.session_state.get('run_analysis_and_plot') and dataframes:
        # Удаляем флаг, чтобы избежать повторного запуска при перезагрузке страницы
        del st.session_state['run_analysis_and_plot']

        with st.spinner("Автоматический запуск симуляции и генерации графиков..."):
            # 1. Объединяем DataFrame
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
                combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
                combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
                st.session_state["combined_df_analysis"] = combined_df
            except Exception as e:
                st.error(f"Ошибка при объединении данных: {str(e)}")
                st.stop() # Прерываем выполнение, если данные не удалось подготовить

            # 2. Собираем параметры и запускаем симуляцию
            params = get_strategy_parameters("analysis")
            simulation_params = params.copy()
            simulation_params["position_size"] = position_size
            simulation_params["commission"] = commission
            simulation_params["aggressive_mode"] = st.session_state.get("aggressive_mode_analysis", False)
            
            simulation_results = run_trading_simulation(combined_df, simulation_params)
            
            # 3. Сохраняем результаты в session_state для отображения ниже
            st.session_state["simulation_results"] = simulation_results
            st.session_state["simulation_params"] = simulation_params
            
            # 4. Устанавливаем флаги для автоматической генерации графиков
            st.session_state['auto_generate_ml_plot'] = True
            st.session_state['auto_generate_screenshots'] = True

        st.success("Автоматический анализ завершен!")
        # st.rerun() # Перезапускаем, чтобы UI обновился и отобразил результаты
    
    if dataframes:
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

            # --- Валидация: все файлы должны относиться к одному инструменту ---
            unique_symbols = combined_df['Symbol'].unique()
            if len(unique_symbols) > 1:
                st.error(
                    f"Ошибка: Выбраны файлы для нескольких инструментов ({', '.join(unique_symbols)}). "
                    "Для корректного анализа и симуляции все файлы должны относиться к одному инструменту."
                )
                st.session_state["combined_df_analysis"] = None # Сбрасываем DataFrame
                combined_df = None

            st.session_state["combined_df_analysis"] = combined_df
        except Exception as e:
            st.error(f"Ошибка при объединении данных: {str(e)}")
            combined_df = None
            st.session_state["combined_df_analysis"] = None
            
        if combined_df is not None:
            with st.spinner("Расчет количества сигналов..."):
                try:
                    # Собираем параметры для генерации сигналов
                    params_for_signals = get_strategy_parameters("analysis")
                    signal_indices, _, _ = generate_signals(combined_df, params_for_signals)
                    st.info(f"Найдено сигналов по текущим параметрам: **{len(signal_indices)}**")
                except Exception as e:
                    st.error(f"Ошибка при генерации сигналов: {e}")
                    signal_indices = []
        
        if st.button("Запустить симуляцию торговли", key="run_simulation"):
            with st.spinner("Запуск симуляции..."):
                # Собираем все параметры из session_state перед запуском
                params = get_strategy_parameters("analysis")

                # Создаем копию для симуляции и добавляем специфичные для нее параметры
                simulation_params = params.copy()
                simulation_params["position_size"] = position_size
                simulation_params["commission"] = commission
                simulation_params["aggressive_mode"] = st.session_state.get("aggressive_mode_analysis", False)
                
                simulation_results = run_trading_simulation(combined_df, simulation_params)
                
                st.session_state["simulation_results"] = simulation_results
                st.session_state["simulation_params"] = simulation_params
                
                st.success("Симуляция завершена!")
        
        if "simulation_results" in st.session_state:
            results = st.session_state["simulation_results"]
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Всего сделок", results['total_trades'])
            col2.metric("Прибыльных сделок", results['winning_trades'])
            col3.metric("Процент прибыльных", f"{round(results['win_rate']*100, 2):.2f}%")
            col4.metric("Общий PnL", f"${results['total_pnl']:.2f}")
            col5.metric("Средний PnL", f"${results['avg_pnl']:.2f}")
            col6.metric("Макс. просадка", f"{round(results['max_drawdown']*100, 2):.2f}%")

            # Добавляем expander для отображения параметров симуляции
            with st.expander("Показать параметры использованной симуляции"):
                sim_params = st.session_state.get("simulation_params", {})
                if sim_params:
                    used_params_list = results.get('used_params', [])
                    # Преобразуем словарь в DataFrame для красивого отображения
                    params_list = []
                    for param, value in sim_params.items():
                        is_used = "✅" if param in used_params_list else "—"
                        params_list.append({"Параметр": param, "Значение": value, "Участвовал в расчетах": is_used})
                    params_df = pd.DataFrame(params_list)
                    params_df['Значение'] = params_df['Значение'].astype(str)  # type: ignore
                    st.dataframe(params_df, use_container_width=True)
                else:
                    st.info("Параметры для этой симуляции не сохранены.")
            
            # 2. Если это была симуляция с ML, дополнительно показываем график анализа модели.
            if results.get('is_ml_simulation'):
                st.subheader("Анализ работы ML-модели")
                st.info("График разделения сигналов по двум наиболее важным признакам.")
                with st.spinner("Генерация ML-графика..."):
                    ml_fig = plot_ml_decision_boundary(combined_df, results, st.session_state.get("simulation_params", {}))
                    if ml_fig:
                        st.plotly_chart(ml_fig, width='stretch')
                    else:
                        st.info("Не удалось построить график анализа ML. Возможно, недостаточно данных или признаков.")

            # --- ПЕРЕРАБОТАННЫЙ БЛОК: Принудительная генерация ML-графика ---
            # Этот блок позволяет построить ML-график на основе текущих параметров в сайдбаре,
            # даже если основная симуляция была запущена без ML.
            st.subheader("Принудительный анализ с ML")
            
            # Проверяем, доступны ли параметры ML (например, загружены из профиля)
            ml_params_available = st.session_state.get("classifier_type_analysis") is not None

            if not ml_params_available:
                st.info("Чтобы построить ML-график, загрузите профиль с ML-параметрами из 'Аналитики' или задайте их вручную.")
            
            # Проверяем, была ли нажата кнопка или установлен флаг авто-генерации
            run_ml_plot = st.button("Построить ML-график", key="force_ml_plot", disabled=not (ml_params_available and combined_df is not None))
            if (run_ml_plot or st.session_state.get('auto_generate_ml_plot')) and ml_params_available and combined_df is not None:
                st.session_state.pop('auto_generate_ml_plot', None) # Удаляем флаг
                with st.spinner("Выполняется принудительный ML-анализ..."):
                    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
                    try:
                        current_params = get_strategy_parameters("analysis")
                        df = st.session_state.get("combined_df_analysis")
                        
                        # Вызываем новую, вынесенную функцию
                        fig = run_forced_ml_analysis(df, current_params)

                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Не удалось построить ML-график. Проверьте консоль или параметры.")
                    except Exception as e:
                        st.error(f"Ошибка при построении ML-графика: {e}")

            if results['trades']:
                st.subheader("Список сделок")
                trades_data = {
                    'Индекс входа': [trade['entry_idx'] for trade in results['trades']],
                    'Цена входа': [trade['entry_price'] for trade in results['trades']],
                    'Направление': [trade['direction'] for trade in results['trades']],
                    'Цена выхода': [trade['exit_price'] for trade in results['trades']],
                    'PnL': [trade['pnl'] for trade in results['trades']],
                    'Причина выхода': [trade['exit_reason'] for trade in results['trades']]
                }
                trades_df_display = pd.DataFrame(trades_data)
                st.dataframe(trades_df_display, use_container_width=True)

            # Новая секция для скриншотов сделок
            if results['trades']:
                st.subheader("Скриншоты сделок")
                # Проверяем, была ли нажата кнопка или установлен флаг авто-генерации
                run_screenshots = st.button("Сгенерировать скриншоты сделок", key="generate_screenshots")
                if run_screenshots or st.session_state.get('auto_generate_screenshots'):
                    st.session_state.pop('auto_generate_screenshots', None) # Удаляем флаг
                    simulation_params = st.session_state.get("simulation_params", {})
                    with st.spinner("Создание скриншотов..."):
                        # Используем до 3 колонок для отображения
                        cols = st.columns(3)
                        for i, trade in enumerate(results['trades']):
                            with cols[i % 3]:
                                trade_fig = plot_single_trade(combined_df, trade, window_size=50, params=st.session_state.get("simulation_params", {}))
                                st.plotly_chart(trade_fig, use_container_width=True)