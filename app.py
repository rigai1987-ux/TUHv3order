import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from datetime import datetime
import numpy as np

# Импортируем функции для каждой страницы
from ui.analysis_page import show_analysis_page
from ui.optimization_page import show_optimization_page
# Заглушка для будущего модуля
# from ui.analytics_page import show_analytics_page

# Импортируем общие утилиты
from app_utils import load_profile_to_session_state

# Импортируем бэкенд модули
import wfo_optimizer # Импортируем новый модуль WFO
import optuna_optimizer as wfo_optuna # Импортируем модуль оптимизации
import strategy_objectives # Импортируем модуль с дополнительными целевыми функциями
# Настройка страницы
st.set_page_config(
    page_title="Streamlit Backtester",
    page_icon="📈",
    layout="wide"
)

# Создание директорий для хранения профилей и прогонов
os.makedirs("profiles", exist_ok=True)
os.makedirs("plots", exist_ok=True) # Директория для графиков
os.makedirs("optimization_runs", exist_ok=True)

# Заголовок приложения
st.title("📈 Streamlit Backtester")

# Навигация
st.sidebar.header("Навигация")

# Создаем три кнопки в одной строке для навигации с подсветкой активной
col1, col2, col3 = st.sidebar.columns(3)

# Устанавливаем текущую страницу из session_state или по умолчанию
current_page = st.session_state.get("page", "Аналитика")

with col1:
    # Определяем, является ли эта кнопка активной
    is_active = current_page == "Анализ сигналов"
    btn_type = "secondary" if not is_active else "primary"
    if st.button("Анализ", key="nav_analyze", type=btn_type, on_click=lambda: st.session_state.update(page="Анализ сигналов")):
        st.session_state.page = "Анализ сигналов"
        st.rerun()

with col2:
    # Определяем, является ли эта кнопка активной
    is_active = current_page == "Оптимизация"
    btn_type = "secondary" if not is_active else "primary"
    if st.button("Оптимизация", key="nav_optimize", type=btn_type, on_click=lambda: st.session_state.update(page="Оптимизация")):
        st.session_state.page = "Оптимизация"
        st.rerun()

with col3:
    # Определяем, является ли эта кнопка активной
    is_active = current_page == "Аналитика"
    btn_type = "secondary" if not is_active else "primary"
    if st.button("Аналитика", key="nav_analytics", type=btn_type, on_click=lambda: st.session_state.update(page="Аналитика")):
        st.session_state.page = "Аналитика"
        st.rerun()

# Функция для обработки загрузки профиля в session_state
def handle_profile_loading():
    """Обработка загрузки профиля в session_state"""
    # Обработка загрузки профиля
    if "profile_to_load" in st.session_state:
        profile_data = st.session_state["profile_to_load"]
        page_to_rerun = st.session_state.get("page_to_rerun")
        
        # Удаляем флаги из session_state до загрузки данных, чтобы избежать конфликта с виджетами
        del st.session_state["profile_to_load"]
        if "page_to_rerun" in st.session_state:
            del st.session_state["page_to_rerun"]
        
        # Очищаем session_state перед загрузкой новых данных, чтобы избежать конфликтов
        # Сохраняем только навигационную информацию
        page_nav = st.session_state.get("page")
        st.session_state.clear()
        st.session_state["page"] = page_nav

        if page_to_rerun == "Анализ сигналов":
            # Загружаем значения в session_state с правильными ключами
            load_profile_to_session_state(profile_data, "analysis")
            
            # Обработка специфичных для анализа параметров
            if "hldir_window" in profile_data:
                st.session_state["hldir_window_analysis"] = profile_data["hldir_window"]
            
            profile_name = profile_data.get('name', 'неизвестный')
            st.success(f"Профиль '{profile_name}' загружен!")
            
            # Если это временный профиль, созданный из оптимизации, удаляем его
            if profile_name.startswith("temp_analysis_"):
                try:
                    import os
                    profile_path = os.path.join("profiles", "analysis", f"{profile_name}.json")
                    if os.path.exists(profile_path):
                        os.remove(profile_path)
                except Exception as e:
                    st.warning(f"Не удалось удалить временный профиль: {str(e)}")
                
        elif page_to_rerun == "Оптимизация":
            # Загружаем значения в session_state с правильными ключами
            load_profile_to_session_state(profile_data, "optimization")
            
            # Обработка специфичных для оптимизации параметров
            if "hldir_window_min" in profile_data:
                st.session_state["hldir_window_min_optimization"] = profile_data["hldir_window_min"]
            if "hldir_window_max" in profile_data:
                st.session_state["hldir_window_max_optimization"] = profile_data["hldir_window_max"]
            
            # Загружаем параметры климаксного выхода
            if profile_data.get("use_climax_exit") == ("categorical", [True]):
                st.session_state["use_climax_exit_option"] = "Да"

            
            st.success(f"Диапазоны '{profile_data.get('name', 'неизвестный')}' загружены!")
        else:
            # Неизвестная страница, выходим
            return
            
        # Перезапускаем страницу для обновления значений виджетов
        st.rerun()

# Основная логика приложения
if current_page == "Анализ сигналов":
    handle_profile_loading() # Эта функция останется в app.py, так как она управляет состоянием
    show_analysis_page()

elif current_page == "Оптимизация":
    handle_profile_loading()
    show_optimization_page()

elif current_page == "Аналитика":
    # Отображение списка сохранённых прогонов оптимизации
    st.header("Аналитика результатов оптимизации")
    st.subheader("Сохранённые прогоны оптимизации")
    
    # Локальный импорт, чтобы избежать циклической зависимости
    from app_utils import get_optimization_run_files, load_run_data_cached
    
    run_files = get_optimization_run_files()
    
    if run_files:
        # Отображаем все прогоны в компактном виде с кнопками в ряд
        for run_file in run_files:
            run_name = run_file.replace('.json', '')
            
            # Загружаем данные прогона для отображения метрик
            run_data = load_run_data_cached(run_file)
            if run_data is None:
                continue
            
            # Создаем контейнер для компактного отображения
            with st.container():
                # Используем columns для компактного размещения информации и кнопок
                cols = st.columns([1, 8])  # Распределяем ширину: кнопка раскрытия, название
                
                with cols[0]:
                    # Кнопка для отображения/скрытия таблицы результатов
                    show_results_key = f"show_results_{run_name}"
                    
                    # Создаем кнопку и проверяем, была ли она нажата
                    button_label = "▼" if st.session_state.get(show_results_key, False) else "▶"
                    if st.button(button_label, key=show_results_key + "_button"):
                        # Изменяем состояние при нажатии кнопки
                        st.session_state[show_results_key] = not st.session_state.get(show_results_key, False)
                        st.rerun() # Перезапускаем, чтобы обновить интерфейс
                    
                with cols[1]:
                    st.markdown(f"**{run_name}**")
            
            # Отображаем таблицу результатов, если она запрошена
            if st.session_state.get(show_results_key, False):
                # Проверяем, есть ли данные в top_10_results (для Optuna) или в results (для WFO)
                if run_data is not None:
                    # Определяем, какие данные использовать для таблицы
                    if "top_10_results" in run_data and run_data["top_10_results"]:
                        # Для Optuna результатов используем top_10_results
                        results_data = run_data["top_10_results"]
                        # Добавляем ID к каждому результату, если его нет
                        for i, result in enumerate(results_data):
                            if 'ID' not in result:
                                result['ID'] = i + 1
                        results_df = pd.DataFrame(results_data)
                    elif "results" in run_data and run_data["results"]:
                        # Для WFO результатов используем results
                        results_df = pd.DataFrame(run_data["results"])
                    else:
                        st.warning("Нет данных для отображения")
                    
                    # --- Отображене кнопок и метрик только для раскрытого элемента ---
                    with st.container():
                        action_cols = st.columns([2, 2, 4])
                        with action_cols[0]:
                            if st.button(f"→ Оптимизация", key=f"optimizer_{run_name}"):
                                optimization_data = {**run_data.get("ranges", {}), **run_data.get("settings", {})}
                                load_profile_to_session_state(optimization_data, "optimization")
                                st.session_state["page"] = "Оптимизация"
                                st.rerun()
                        
                        with action_cols[1]:
                            # Кнопка для перехода в анализ с лучшими параметрами
                            if st.button(f"→ Анализ (лучший)", key=f"analysis_from_run_{run_name}"):
                                best_params = run_data.get("best_params", {})
                                # Собираем все данные: лучшие параметры + базовые настройки
                                analysis_data = {**run_data.get("settings", {}), **best_params}
                                
                                # Добавляем файлы данных, использованные в этом прогоне
                                if "data_files" in run_data:
                                    analysis_data["selected_files"] = run_data["data_files"]
                                
                                # --- ЛОГИРОВАНИЕ: Выводим передаваемые параметры в консоль ---
                                import pprint
                                print(f"\n[LOG] Передача параметров из 'Аналитики' -> 'Анализ' (лучшие параметры рана '{run_name}'):")
                                pprint.pprint(analysis_data)
                                print("-" * 70)
                                # --- Конец блока логирования ---
                                load_profile_to_session_state(analysis_data, "analysis") # Загружаем в session_state
                                st.session_state["page"] = "Анализ сигналов"
                                st.rerun()

                        with action_cols[2]:
                            # Извлечение и отображение метрик
                            top_results = run_data.get("top_10_results", [])
                            if top_results:
                                best_result = top_results[0]
                                total_pnl = best_result.get('PnL', 0) or best_result.get('total_pnl', 0)
                                if total_pnl is None or (isinstance(total_pnl, float) and np.isnan(total_pnl)):
                                    pnl = 'N/A'
                                else:
                                    pnl = f"${total_pnl:.2f}" if isinstance(total_pnl, (int, float)) else str(total_pnl)

                                win_rate = best_result.get('Win Rate', 0) or best_result.get('win_rate', 0)
                                if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
                                    win_rate_formatted = 'N/A'
                                elif isinstance(win_rate, (int, float)):
                                    win_rate_formatted = f"{round(win_rate * 100, 2) if win_rate <= 1 else win_rate:.2f}%"
                                else:
                                    win_rate_formatted = str(win_rate)
                            else:
                                best_result = run_data.get("best_result", {})
                                total_pnl = best_result.get('total_pnl', 0)
                                if total_pnl is None or (isinstance(total_pnl, float) and np.isnan(total_pnl)):
                                    pnl = 'N/A'
                                else:
                                    pnl = f"${total_pnl:.2f}"

                                win_rate = best_result.get('win_rate', 0)
                                if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
                                    win_rate_formatted = 'N/A'
                                else:
                                    win_rate_formatted = f"{round(win_rate * 100, 2) if win_rate <= 1 else win_rate:.2f}%"

                            st.caption(f"Лучший результат: PnL: {pnl}, WR: {win_rate_formatted}")

                    # --- Конец блока кнопок и метрик ---

                    # Переупорядочиваем столбцы в соответствии с порядком в результатах оптимизации
                    desired_order = [
                        "ID",
                        "Total Trades", "PnL", "Win Rate", "Max Drawdown", "Sharpe Ratio", "Profit Factor",
                        "vol_pctl", "vol_period", "rng_pctl", "range_period", "natr_min", "natr_period",
                        "min_growth_pct", "lookback_period", "prints_analysis_period", "prints_threshold_ratio",
                        "stop_loss_pct", "take_profit_pct"
                    ]
                    # Добавляем параметры "вилки", если они могут быть в результатах
                    desired_order.extend(["bracket_offset_pct", "bracket_timeout_candles"])
                    
                    # Убедимся, что все столбцы из desired_order присутствуют в DataFrame
                    available_columns = [col for col in desired_order if col in results_df.columns]
                    # Добавим любые столбцы, которые могут отсутствовать в desired_order, в конец, кроме 'value'
                    additional_columns = [col for col in results_df.columns if col not in desired_order]
                    final_order = available_columns + additional_columns
                    
                    results_df_display = results_df[final_order].copy()
                    
                    # Отображаем параметры (все результаты теперь используют режим 1)
                    display_df = results_df_display.copy()
                    
                    # Добавляем колонку с кнопками для выбора результата
                    selected_result_key = f"selected_result_{run_name}"
                    # Добавляем еще одну колонку для кнопки "Графики"
                    cols = st.columns([1, 1, 8])  # Колонки для кнопок "В анализ", "Графики" и основной таблицы
                    with cols[0]:
                        st.write("**В анализ**")
                        for i in range(len(results_df)):
                            # Получаем информацию о результате для отображения на кнопке
                            result_row = results_df.iloc[i]
                            result_id = result_row.get('ID', i+1)
                            # Убираем PnL и WR с кнопки для компактности
                            pnl = result_row.get('PnL', 'N/A')
                            win_rate = result_row.get('Win Rate', 'N/A')
                            
                            # Кнопка с информацией о результате (только значения), которая сразу переходит в анализ
                            if st.button(f"Парам. {result_id}", key=f"select_{run_name}_result_{i}", help="Загрузить параметры этого результата на страницу 'Анализ'"):
                                # Загружаем параметры выбранного результата в session_state с правильными ключами
                                selected_params_from_row = {k: v for k, v in result_row.items() if k != 'ID'}
                                # Собираем все данные: базовые настройки из всего прогона + параметры из строки.
                                # Параметры из строки (selected_params_from_row) должны иметь приоритет,
                                # поэтому они идут вторыми при слиянии.
                                full_params_to_load = {**run_data.get("settings", {}), **selected_params_from_row}

                                # Добавляем файлы данных, использованные в этом прогоне
                                if "data_files" in run_data:
                                    full_params_to_load["selected_files"] = run_data["data_files"]
                                
                                # --- ЛОГИРОВАНИЕ: Выводим передаваемые параметры в консоль ---
                                import pprint
                                print(f"\n[LOG] Передача параметров из 'Аналитики' -> 'Анализ' (параметры из строки ID {result_id}):")
                                pprint.pprint(full_params_to_load)
                                print("-" * 70)
                                # --- Конец блока логирования ---
                                    
                                load_profile_to_session_state(full_params_to_load, "analysis")
                                
                                st.session_state["page"] = "Анализ сигналов"
                                st.rerun()
                                
                    with cols[1]:
                        st.write("**Графики**")
                        for i in range(len(results_df)):
                            result_row = results_df.iloc[i]
                            result_id = result_row.get('ID', i+1)
                            # Новая кнопка для генерации графиков
                            if st.button(f"📊 {result_id}", key=f"plot_{run_name}_result_{i}", help="Перейти в 'Анализ' и автоматически сгенерировать все графики для этого результата"):
                                # Собираем параметры так же, как и для кнопки "В анализ"
                                selected_params_from_row = {k: v for k, v in result_row.items() if k != 'ID'}
                                full_params_to_load = {**run_data.get("settings", {}), **selected_params_from_row}

                                if "data_files" in run_data:
                                    full_params_to_load["selected_files"] = run_data["data_files"]

                                # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Добавляем флаг для авто-запуска ---
                                full_params_to_load['run_analysis_and_plot'] = True

                                # Загружаем параметры и флаг в session_state
                                load_profile_to_session_state(full_params_to_load, "analysis")
                                
                                # Переходим на страницу анализа и перезапускаем
                                st.session_state["page"] = "Анализ сигналов"
                                st.rerun()

                                
                    with cols[2]:
                        # Отображаем таблицу с корректным отображением параметров в зависимости от режима анализа принтов
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("Нет данных для отображения")
            
            st.markdown("---")
    else:
        st.info("Нет сохранённых прогонов оптимизации")