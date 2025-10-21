import streamlit as st
import pandas as pd
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff

from signal_generator import generate_signals
from trading_simulator import run_trading_simulation
from visualizer import plot_single_trade, plot_ml_decision_boundary, plot_wfo_results
from ml_model_handler import train_ml_model, label_all_signals, generate_features
from app_utils import get_basic_settings, manage_profiles, get_strategy_parameters, load_and_validate_data_files # type: ignore

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
        st.checkbox(
            "Симулировать проскальзывание",
            value=st.session_state.get("simulate_slippage_analysis", True),
            key="simulate_slippage_analysis",
            help="Если включено, цена входа может быть хуже уровня 'вилки' в случае гэпа (более реалистично). Если отключено, вход всегда происходит точно по уровню."
        )


        # --- НОВЫЙ БЛОК: Параметры для ML Классификатора ---
        st.subheader("🤖 ML Классификатор")
        st.number_input(
            "Тайм-аут для разметки (свечи)",
            value=int(st.session_state.get("ml_labeling_timeout_candles_analysis", 10)),
            min_value=1, step=1, key="ml_labeling_timeout_candles_analysis",
            help="Сколько свечей смотреть в будущее для определения успеха/провала сигнала."
        )
        st.number_input(
            "Период для ML-признаков",
            value=int(st.session_state.get("ml_prints_window_analysis", 10)),
            min_value=1, step=1, key="ml_prints_window_analysis",
            help="Размер скользящего окна для признаков 'prints_strength', 'market_strength' и 'hldir_strength'."
        )
        st.number_input("Итерации (деревья)", value=int(st.session_state.get("ml_iterations_analysis", 300)), min_value=10, step=10, key="ml_iterations_analysis")
        st.number_input("Глубина деревьев", value=int(st.session_state.get("ml_depth_analysis", 4)), min_value=2, max_value=10, step=1, key="ml_depth_analysis")
        st.number_input("Скорость обучения", value=float(st.session_state.get("ml_learning_rate_analysis", 0.1)), min_value=0.01, step=0.01, format="%.2f", key="ml_learning_rate_analysis")
        
        st.session_state['use_ml_filter_analysis'] = st.toggle(
            "Использовать ML-фильтр",
            value=st.session_state.get('use_ml_filter_analysis', False),
            help="Если включено, симулятор будет использовать обученную ML-модель для фильтрации сигналов."
        )
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

            # Передаем ML модель в симулятор, если она обучена и фильтр включен
            if st.session_state.get('use_ml_filter_analysis') and 'ml_model_bundle' in st.session_state:
                simulation_params['ml_model_bundle'] = st.session_state['ml_model_bundle']
            else:
                simulation_params.pop('ml_model_bundle', None) # Убираем, если не используется
            
            simulation_results = run_trading_simulation(combined_df, simulation_params)
            
            # 3. Сохраняем результаты в session_state для отображения ниже
            st.session_state["simulation_results"] = simulation_results
            st.session_state["simulation_params"] = simulation_params # Сохраняем параметры, с которыми была симуляция
            
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

                # Передаем ML модель в симулятор, если она обучена и фильтр включен
                if st.session_state.get('use_ml_filter_analysis') and 'ml_model_bundle' in st.session_state:
                    simulation_params['ml_model_bundle'] = st.session_state['ml_model_bundle']
                else:
                    simulation_params.pop('ml_model_bundle', None) # Убираем, если не используется
                
                simulation_results = run_trading_simulation(combined_df, simulation_params)
                
                st.session_state["simulation_results"] = simulation_results
                st.session_state["simulation_params"] = simulation_params # Сохраняем параметры, с которыми была симуляция

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

            # --- НОВЫЙ БЛОК: Обучение ML модели ---
            st.subheader("Обучение ML-фильтра")
            if st.button("Подготовить данные и обучить модель", key="train_ml_model"):
                with st.spinner("Шаг 1/3: Генерация признаков..."):
                    params = get_strategy_parameters("analysis")
                    # Генерируем сигналы и все индикаторы
                    signal_indices, df_with_indicators, _ = generate_signals(combined_df, params, return_indicators=True)
                    # Обогащаем DataFrame ML-признаками
                    df_with_features = generate_features(df_with_indicators, params)
                
                if df_with_features is not None and signal_indices:
                    with st.spinner("Шаг 2/3: Разметка всех сигналов (создание X и y)..."):
                        # Используем новую функцию для разметки ВСЕХ сигналов
                        X, y = label_all_signals(df_with_features, signal_indices, params)
                    
                    if not X.empty and not y.empty:
                        st.info(f"Для обучения найдено {len(y)} сигналов ({y.sum()} успешных, {len(y) - y.sum()} провальных).")
                        
                        # --- УЛУЧШЕНИЕ: Разделение данных на обучающую и тестовую выборки ---
                        st.markdown("#### Настройки обучения и валидации")
                        test_size_pct = st.slider(
                            "Размер тестовой выборки (%)", 
                            min_value=10, max_value=50, value=30, step=5, 
                            key="ml_test_size_pct",
                            help="Процент данных, который будет отложен для тестирования модели после обучения. Обучение будет проводиться на оставшихся данных."
                        )
                        test_size = test_size_pct / 100.0

                        # Разделяем данные, сохраняя временной порядок (shuffle=False)
                        X_train, X_test, y_train, y_test = train_test_split( # type: ignore
                            X, y, test_size=test_size, shuffle=False
                        )
                        st.write(f"Данные разделены: **{len(y_train)}** сигналов для обучения, **{len(y_test)}** для теста.")
                        # --- Конец блока ---

                        with st.spinner(f"Шаг 3/3: Обучение модели CatBoost на {len(y_train)} сигналах..."):
                            ml_params = get_strategy_parameters("analysis")
                            # --- ИЗМЕНЕНИЕ: Обучаем модель только на обучающей выборке ---
                            model_bundle = train_ml_model(X_train, y_train, ml_params)
                        
                        # Сохраняем обученную модель в session_state
                        st.session_state['ml_model_bundle'] = model_bundle
                        # --- ИЗМЕНЕНИЕ: Сохраняем обучающие данные для визуализации ---
                        st.session_state['ml_training_data'] = (X_train, y_train) # type: ignore
                        # Сохраняем тестовые данные для оценки
                        st.session_state['ml_test_data'] = (X_test, y_test) # type: ignore

                        st.success("Модель успешно обучена и сохранена!")
                        
                        # --- НОВЫЙ БЛОК: Оценка модели на тестовой выборке ---
                        if not X_test.empty:
                            st.subheader("Оценка производительности модели на тестовых данных")
                            with st.spinner("Оценка на тестовой выборке..."):
                                # Извлекаем компоненты
                                model = model_bundle['model']
                                scaler = model_bundle['scaler']
                                feature_names = model_bundle['feature_names']
                                numerical_features = model_bundle['numerical_features']
                                
                                # Масштабируем тестовые данные тем же скейлером, что и обучающие
                                X_test_scaled = X_test.copy() # type: ignore
                                # Проверяем, что есть числовые признаки для масштабирования
                                num_features_in_df = [f for f in numerical_features if f in X_test_scaled.columns]
                                if scaler and num_features_in_df:
                                    X_test_scaled[num_features_in_df] = scaler.transform(X_test_scaled[num_features_in_df])
                                
                                # Делаем предсказания
                                y_pred = model.predict(X_test_scaled[feature_names]) # type: ignore
                                
                                y_test_series = y_test # type: ignore
                                # Считаем и выводим метрики
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{accuracy_score(y_test_series, y_pred):.2%}")
                                col2.metric("Precision", f"{precision_score(y_test_series, y_pred, zero_division=0):.2%}")
                                col3.metric("Recall", f"{recall_score(y_test_series, y_pred, zero_division=0):.2%}")
                                col4.metric("F1-Score", f"{f1_score(y_test_series, y_pred, zero_division=0):.2%}")

                                # Строим матрицу ошибок
                                cm = confusion_matrix(y_test_series, y_pred)
                                cm_fig = ff.create_annotated_heatmap(
                                    z=cm, x=['Предсказано 0', 'Предсказано 1'], y=['Реально 0', 'Реально 1'], 
                                    colorscale='Blues', showscale=True
                                )
                                cm_fig.update_layout(title_text='Матрица ошибок (Confusion Matrix)')
                                st.plotly_chart(cm_fig, use_container_width=True)
                        # --- Конец блока оценки ---

                        # Отображаем важность признаков
                        st.subheader("Важность признаков (Feature Importances)")
                        df_to_display = model_bundle['feature_importances'].copy()
                        df_to_display = df_to_display.rename(columns={'feature': 'Признак', 'importance': 'Важность (%)', 'description': 'Описание и формула'})
                        st.dataframe(df_to_display)

                        # --- НОВЫЙ БЛОК: Визуализация границ решений ---
                        st.subheader("Визуализация границ решений")
                        # --- ИЗМЕНЕНИЕ: Автоматическая генерация графиков для пар топ-признаков ---
                        all_features = model_bundle['feature_names'] # type: ignore
                        if len(all_features) >= 2:
                            import itertools

                            # Виджет для выбора количества топ-признаков
                            num_top_features = st.slider(
                                "Количество топ-признаков для анализа пар",
                                min_value=2,
                                max_value=min(len(all_features), 10), # Ограничиваем 7, чтобы избежать слишком большого кол-ва графиков
                                value=min(len(all_features), 10), # По умолчанию 4
                                step=1,
                                help="Выберите, сколько самых важных признаков использовать для построения графиков парных взаимодействий."
                            )

                            # Получаем список топ-N признаков
                            top_features = model_bundle['feature_importances']['feature'].head(num_top_features).tolist()

                            # Генерируем все уникальные пары из этих признаков
                            feature_pairs = list(itertools.combinations(top_features, 2))

                            st.info(f"Будет построено **{len(feature_pairs)}** графиков для топ-{num_top_features} признаков.")

                            # Отображаем графики в 2 колонки
                            cols = st.columns(2)
                            for i, (feature1, feature2) in enumerate(feature_pairs):
                                with cols[i % 2]:
                                    with st.spinner(f"Построение графика для '{feature1}' и '{feature2}'..."):
                                        # --- ИЗМЕНЕНИЕ: Строим график на обучающих данных ---
                                        fig = plot_ml_decision_boundary(model_bundle, X_train, y_train, feature1, feature2)
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Недостаточно признаков для построения 2D-графика границ решений.")

                    else:
                        st.warning("Не удалось найти достаточно данных для разметки и обучения. Попробуйте изменить параметры разметки или стратегии.")
                else:
                    st.error("Не удалось сгенерировать признаки для обучения. Проверьте параметры стратегии.")

            if results['trades']:
                # --- ИЗМЕНЕНИЕ: Разделяем сделки для корректного отображения в таблице ---
                real_trades = [t for t in results['trades'] if not t.get('skipped_by_ml')]
                st.subheader(f"Список реальных сделок ({len(real_trades)})")
                trades_data = {
                    'Индекс входа': [trade['entry_idx'] for trade in real_trades],
                    'Цена входа': [trade['entry_price'] for trade in real_trades],
                    'Направление': [trade['direction'] for trade in real_trades],
                    'Цена выхода': [trade['exit_price'] for trade in real_trades],
                    'PnL': [trade['pnl'] for trade in real_trades],
                    'Причина выхода': [trade['exit_reason'] for trade in real_trades]
                }
                trades_df_display = pd.DataFrame(trades_data)
                st.dataframe(trades_df_display, use_container_width=True)

            # --- ИЗМЕНЕНИЕ: Секция для скриншотов теперь отображает ВСЕ сделки (реальные и пропущенные) ---
            if results['trades']:
                st.subheader("Хронология сделок (включая пропущенные ML-фильтром)")
                # Сортируем все сделки по индексу входа для правильной хронологии
                all_trades_sorted = sorted(results['trades'], key=lambda x: x['entry_idx'])
                # Проверяем, была ли нажата кнопка или установлен флаг авто-генерации
                run_screenshots = st.button("Сгенерировать скриншоты сделок", key="generate_screenshots")
                if run_screenshots or st.session_state.get('auto_generate_screenshots'):
                    st.session_state.pop('auto_generate_screenshots', None) # Удаляем флаг
                    simulation_params = st.session_state.get("simulation_params", {})
                    with st.spinner(f"Создание {len(all_trades_sorted)} скриншотов..."):
                        cols = st.columns(3) # Используем до 3 колонок для отображения
                        for i, trade in enumerate(all_trades_sorted):
                            with cols[i % 3]:
                                trade_fig = plot_single_trade(combined_df, trade, window_size=50, params=simulation_params)
                                st.plotly_chart(trade_fig, use_container_width=True)