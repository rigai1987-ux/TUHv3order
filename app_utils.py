import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import re
import tempfile

def _sanitize_filename(filename: str) -> str:
    """
    Очищает имя файла от недопустимых символов.
    """
    # Удаляем символы, запрещенные в именах файлов Windows и других ОС
    return re.sub(r'[<>:"/\\|?*]', '', filename).strip()

def _atomic_write(file_path: str, data: str):
    """
    Атомарно записывает данные в файл.
    Сначала пишет во временный файл, затем переименовывает его.
    Это предотвращает порчу файла при сбое во время записи.
    """
    # Создаем временный файл в той же директории, чтобы `os.rename` был атомарным
    temp_dir = os.path.dirname(file_path)
    # `delete=False` нужно для Windows, чтобы можно было переименовать файл после закрытия
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=temp_dir, delete=False) as tmp:
        tmp.write(data)
        temp_path = tmp.name
    
    # Переименовываем временный файл в целевой. Эта операция обычно атомарна.
    os.replace(temp_path, file_path)

# Функции для работы с профилями
def get_profile_directory(module):
    """Получить директорию для профилей по модулю"""
    # Проверяем, что модуль один из допустимых
    if module not in ["analysis", "optimization"]:
        raise ValueError(f"Неподдерживаемый модуль: {module}")
    return f"profiles/{module}"

def save_profile(profile_name, data, module):
    """Сохранить профиль в JSON-файл"""
    # Валидация входных данных
    if not profile_name or not isinstance(profile_name, str):
        st.error("Название профиля должно быть непустой строкой")
        return False
    
    if not data or not isinstance(data, dict):
        st.error("Данные профиля должны быть непустым словарем")
        return False
    
    try:
        # Очищаем имя профиля от недопустимых символов перед сохранением
        sanitized_profile_name = _sanitize_filename(profile_name)
        directory = get_profile_directory(module)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{sanitized_profile_name}.json")        
        # Сериализуем данные в строку JSON
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        # Используем атомарную запись
        _atomic_write(file_path, json_data)
        
        st.success(f"Профиль '{profile_name}' сохранён!")
        return True
    except (IOError, OSError) as e:
        st.error(f"Ошибка ввода-вывода при сохранении профиля: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Ошибка при сохранении профиля: {str(e)}")
        return False

@st.cache_data(ttl=300)  # Кэшируем на 5 минут
def load_profile(profile_name, module):
    """Загрузить профиль из JSON-файла"""
    # Валидация входных данных
    if not profile_name or not isinstance(profile_name, str):
        st.error("Название профиля должно быть непустой строкой")
        return None
    
    try:
        directory = get_profile_directory(module)
        file_path = os.path.join(directory, f"{profile_name}.json")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Проверяем, что загруженный файл содержит допустимые данные
        if not isinstance(data, dict):
            st.error(f"Файл профиля '{profile_name}' содержит некорректные данные")
            return None
            
        return data
    except FileNotFoundError:
        st.error(f"Профиль '{profile_name}' не найден!")
        return None
    except json.JSONDecodeError:
        st.error(f"Файл профиля '{profile_name}' содержит некорректный JSON")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке профиля: {str(e)}")
        return None

@st.cache_data(ttl=10)  # Кэшируем на 60 секунд
def get_profiles(module):
    """Получить список доступных профилей"""
    try:
        directory = get_profile_directory(module)
        profiles = [f.replace('.json', '') for f in os.listdir(directory) if f.endswith('.json')]
        return profiles
    except FileNotFoundError:
        # Директория может не существовать, если нет профилей
        return []
    except Exception as e:
        st.error(f"Ошибка при получении списка профилей: {str(e)}")
        return []

# Функция для загрузки профиля в session_state
def load_profile_to_session_state(profile_data, module):
    """Загрузить данные профиля в session_state для указанного модуля"""
    # Валидация входных данных
    if not isinstance(profile_data, dict):
        st.error("Данные профиля должны быть словарем.")
        return False
    
    try:
        # Обработка выбранных файлов
        if 'selected_files' in profile_data:
            selected_files_from_profile = profile_data['selected_files']
            
            # Получаем список всех доступных CSV-файлов
            all_csv_files = []
            try:
                all_csv_files = [f for f in os.listdir("dataCSV") if f.endswith(".parquet")]
            except FileNotFoundError:
                st.warning("Папка dataCSV не найдена, не удалось восстановить выбор файлов.")

            # Устанавливаем состояние каждого чекбокса
            for csv_file in all_csv_files:
                is_selected = csv_file in selected_files_from_profile
                st.session_state[f"csv_{module}_{csv_file}"] = is_selected # Ключ должен совпадать с ключом в ui

        for key, value in profile_data.items():
            # Проверяем, что ключ - строка
            if not isinstance(key, str):
                st.warning(f"Пропуск недопустимого ключа: {key}")
                continue
            
            # Преобразуем строку даты в datetime.date, если это дата
            if key in ["start_date", "end_date"] and isinstance(value, str):
                try:
                    # Преобразуем строку в формат datetime.date
                    date_obj = datetime.strptime(value.split()[0], "%Y-%m-%d").date()
                except ValueError:
                    st.warning(f"Не удалось распознать формат даты для '{key}' (значение: '{value}'). Пропускаем.", icon="⚠️")
                    continue # Пропускаем этот ключ, чтобы не записать некорректную строку
                st.session_state[f"{key}_{module}"] = date_obj
            # Обработка категориальных параметров для оптимизации
            elif key == "use_climax_exit" and module == "optimization":
                 if isinstance(value, (list, tuple)) and len(value) > 1 and isinstance(value[1], list):
                    st.session_state["use_climax_exit_option"] = "Да" if value[1][0] else "Нет"
            else:
                # --- УПРОЩЕНИЕ: Единая логика для всех остальных параметров ---
                # Ключ в session_state всегда должен иметь суффикс модуля (например, `_analysis`).
                st.session_state[f"{key}_{module}"] = value
        return True
    except Exception as e:
        st.error(f"Ошибка при загрузке профиля в session_state: {str(e)}")
        return False
def get_strategy_parameters(module):
    """
    Собирает параметры стратегии для анализа из session_state.
    Не создает виджеты, только собирает данные.
    """
    params = {
        "vol_period": st.session_state.get(f"vol_period_{module}", 20),
        "vol_pctl": st.session_state.get(f"vol_pctl_{module}", 1.0),
        "range_period": st.session_state.get(f"range_period_{module}", 20),
        "rng_pctl": st.session_state.get(f"rng_pctl_{module}", 1.0),
        "entry_logic_mode": "Вилка отложенных ордеров",
        "natr_period": st.session_state.get(f"natr_period_{module}", 10),
        "natr_min": st.session_state.get(f"natr_min_{module}", 0.35),
        "lookback_period": st.session_state.get(f"lookback_period_{module}", 20),
        "min_growth_pct": st.session_state.get(f"min_growth_pct_{module}", 1.0),
        "stop_loss_pct": st.session_state.get(f"stop_loss_pct_{module}", 2.0),
        "take_profit_pct": st.session_state.get(f"take_profit_pct_{module}", 4.0),
        "use_climax_exit": st.session_state.get(f"use_climax_exit_{module}", False),
        "climax_exit_window": st.session_state.get(f"climax_exit_window_{module}", 50),
        "climax_exit_threshold": st.session_state.get(f"climax_exit_threshold_{module}", 3.0), # type: ignore
        "bracket_offset_pct": st.session_state.get(f"bracket_offset_pct_{module}", 0.5),
        "bracket_timeout_candles": st.session_state.get(f"bracket_timeout_candles_{module}", 5),
        "invert_direction": st.session_state.get(f"invert_direction_{module}", False),
    }

    # --- Улучшенная и унифицированная логика для сбора параметров ML ---
    # Это гарантирует, что параметры будут собраны, даже если виджеты для них
    # не были явно созданы на странице (например, при загрузке из профиля из аналитики).
    # Ключ в итоговом словаре `params` должен быть "чистым", без суффикса `_{module}`.
    ml_param_keys = [
        "classifier_type", "catboost_iterations", "catboost_depth", "catboost_learning_rate",
        "prints_analysis_period", "prints_threshold_ratio", "m_analysis_period",
        "m_threshold_ratio", "hldir_window", "hldir_offset"
    ]

    for key in ml_param_keys:
        session_key = f"{key}_{module}"
        if session_key in st.session_state:
            # Ключ в итоговом словаре `params` должен быть "чистым", без суффикса,
            # так как именно такие ключи ожидают `train_ml_model` и `generate_signals`.
            params[key] = st.session_state[session_key]

    # Удаляем ключи со значением None, чтобы не передавать лишнего
    return {k: v for k, v in params.items() if v is not None}

# Функция для обработки одного файла данных
def process_single_file(file, module=None):
    """Обработка одного CSV-файла"""
    file_path = os.path.join("dataCSV", file)
    try:
        if os.path.exists(file_path):
            # Чтение Parquet файла (замена с CSV)
            df = pd.read_parquet(file_path)
            
            # Проверяем, что файл имеет ожидаемые столбцы
            expected_columns = ['Symbol', 'time', 'open', 'high', 'low', 'close', 'Volume', 'HLdir', 'long_prints', 'short_prints', 'LongM', 'ShortM']
            if not all(col in df.columns for col in expected_columns):
                st.error(f"Файл {file} не содержит ожидаемые столбцы. Найдены: {list(df.columns)}")
                return None
            
            # Переименовываем столбцы в соответствии с ожиданиями остальной части кода
            df = df.rename(columns={'Volume': 'volume'})
            
            # Если все проверки пройдены, возвращаем DataFrame
            return df
        else:
            st.error(f"Файл {file} не найден по пути {file_path}")
            return None
    except Exception as e:
        st.error(f"Не удалось обработать файл {file}: {e}")
        return None

# Функция для загрузки и валидации CSV-файлов
def load_and_validate_data_files(selected_files, module=None):
    """Загрузка и валидация файлов данных (Parquet)."""
    dataframes = []
    for file in selected_files:
        df = process_single_file(file, module)
        if df is not None:
            dataframes.append(df)
    return dataframes

def get_strategy_parameters_old(module):
    """
    Собирает параметры стратегии для анализа из session_state.
    Не создает виджеты, только собирает данные.
    """
    params = {
        "vol_period": st.session_state.get(f"vol_period_{module}", 20),
        "vol_pctl": st.session_state.get(f"vol_pctl_{module}", 1.0),
        "range_period": st.session_state.get(f"range_period_{module}", 20),
        "rng_pctl": st.session_state.get(f"rng_pctl_{module}", 1.0),
        "entry_logic_mode": "Вилка отложенных ордеров",
        "natr_period": st.session_state.get(f"natr_period_{module}", 10),
        "natr_min": st.session_state.get(f"natr_min_{module}", 0.35),
        "lookback_period": st.session_state.get(f"lookback_period_{module}", 20),
        "min_growth_pct": st.session_state.get(f"min_growth_pct_{module}", 1.0),
        "stop_loss_pct": st.session_state.get(f"stop_loss_pct_{module}", 2.0),
        "take_profit_pct": st.session_state.get(f"take_profit_pct_{module}", 4.0),
        "use_climax_exit": st.session_state.get(f"use_climax_exit_{module}", False),
        "climax_exit_window": st.session_state.get(f"climax_exit_window_{module}", 50),
        "climax_exit_threshold": st.session_state.get(f"climax_exit_threshold_{module}", 3.0), # type: ignore
        "bracket_offset_pct": st.session_state.get(f"bracket_offset_pct_{module}", 0.5),
        "bracket_timeout_candles": st.session_state.get(f"bracket_timeout_candles_{module}", 5),
        "invert_direction": st.session_state.get(f"invert_direction_{module}", False),
    }

    # --- Улучшенная и унифицированная логика для сбора параметров ML ---
    # Это гарантирует, что параметры будут собраны, даже если виджеты для них
    # не были явно созданы на странице (например, при загрузке из профиля из аналитики).
    # Ключ в итоговом словаре `params` должен быть "чистым", без суффикса `_{module}`.
    ml_param_keys = [
        "classifier_type", "catboost_iterations", "catboost_depth", "catboost_learning_rate",
        "prints_analysis_period", "prints_threshold_ratio", "m_analysis_period",
        "m_threshold_ratio", "hldir_window", "hldir_offset"
    ]

    for key in ml_param_keys:
        session_key = f"{key}_{module}"
        if session_key in st.session_state:
            # Ключ в итоговом словаре `params` должен быть "чистым", без суффикса,
            # так как именно такие ключи ожидают `train_ml_model` и `generate_signals`.
            params[key] = st.session_state[session_key]

    # Удаляем ключи со значением None, чтобы не передавать лишнего
    return {k: v for k, v in params.items() if v is not None}

# Функция для генерации параметров оптимизации (теперь использует универсальную функцию)
def get_optimization_parameters():
    """Получить параметры оптимизации (диапазоны) для модуля оптимизации"""
    # Эта функция теперь просто собирает диапазоны из UI, которые определены в optimization_page.py
    # Для сохранения совместимости с manage_profiles, она может остаться как заглушка
    # или быть переработана для сбора данных из session_state, как get_strategy_parameters.
    return {} # Основная логика сбора диапазонов находится в get_param_space_from_ui

# Функция для получения базовых настроек (для обоих модулей)
def get_basic_settings(module):
    """Получить базовые настройки для модуля"""
    col1, col2 = st.columns(2)
    with col1:
        # Инициализируем значение в session_state, если его нет
        if f"position_size_{module}" not in st.session_state:
            st.session_state[f"position_size_{module}"] = 100.0
        # Создаем виджет с явным указанием значения из session_state
        position_size = st.number_input("Размер позиции", value=round(float(st.session_state[f"position_size_{module}"]), 2), step=10.0, key=f"position_size_{module}")
    with col2:
        # Инициализируем значение в session_state, если его нет
        if f"commission_{module}" not in st.session_state:
            st.session_state[f"commission_{module}"] = 0.1
        # Создаем виджет с явным указанием значения из session_state
        commission = st.number_input("Комиссия (%)", value=round(float(st.session_state[f"commission_{module}"]), 3), step=0.01, key=f"commission_{module}")
    col1, col2 = st.columns(2)
    with col1:
        # Инициализируем значение в session_state, если его нет
        if f"start_date_{module}" not in st.session_state:
            st.session_state[f"start_date_{module}"] = datetime(2025, 9, 1).date()
        # Создаем виджет с явным указанием значения из session_state
        start_date = st.date_input("Дата начала", value=st.session_state[f"start_date_{module}"], key=f"start_date_{module}")
    with col2:
        # Инициализируем значение в session_state, если его нет
        if f"end_date_{module}" not in st.session_state:
            st.session_state[f"end_date_{module}"] = datetime(2025, 9, 26).date()
        # Создаем виджет с явным указанием значения из session_state
        end_date = st.date_input("Дата окончания", value=st.session_state[f"end_date_{module}"], key=f"end_date_{module}")
    
    # Добавляем переключатель для агрессивного режима
    aggressive_mode = st.checkbox(
        "Агрессивный режим",
        value=st.session_state.get(f"aggressive_mode_{module}", False),
        key=f"aggressive_mode_{module}",
        help="Позволяет открывать новую сделку, не дожидаясь закрытия предыдущей, если появляется новый сигнал."
    )
    return position_size, commission, start_date, end_date

# Функция для управления профилями (для обоих модулей)
def manage_profiles(module, params_func):
    """Управление профилями для указанного модуля"""
    st.subheader("Профили")
    col1, col2 = st.columns(2)
    with col1:
        profile_name = st.text_input("Название профиля", key=f"profile_name_{module}")
        if st.button(f"Сохранить {'профиль' if module == 'analysis' else 'диапазоны'}", key=f"save_profile_{module}"):
            if profile_name:
                # Сбор всех параметров стратегии
                profile_data = {
                    "position_size": st.session_state.get(f"position_size_{module}", 1000.0),
                    "commission": st.session_state.get(f"commission_{module}", 0.1),
                    "entry_logic_mode": "Вилка отложенных ордеров",
                    "aggressive_mode": st.session_state.get(f"aggressive_mode_{module}", False),
                    "start_date": str(st.session_state.get(f"start_date_{module}", datetime(2025, 9, 1))),
                    "end_date": str(st.session_state.get(f"end_date_{module}", datetime(2025, 9, 26)))
                }
                if module == "analysis":
                    # При сохранении профиля анализа, получаем параметры из session_state напрямую
                    profile_data.update({
                        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Добавляем все параметры, включая "скрытые" ML-параметры ---
                        # get_strategy_parameters соберет ВСЕ параметры из session_state, включая те,
                        # что были загружены из аналитики и не имеют активных виджетов.
                        **get_strategy_parameters("analysis")
                    }) # type: ignore
                else: # optimization
                    # При сохранении профиля оптимизации, получаем параметры из session_state
                    profile_data.update({
                        "vol_period_min": st.session_state.get("vol_period_min_optimization", 10),
                        "vol_period_max": st.session_state.get("vol_period_max_optimization", 30),
                        "vol_pctl_min": st.session_state.get("vol_pctl_min_optimization", 1),
                        "vol_pctl_max": st.session_state.get("vol_pctl_max_optimization", 5),
                        "range_period_min": st.session_state.get("range_period_min_optimization", 10),
                        "range_period_max": st.session_state.get("range_period_max_optimization", 30),
                        "rng_pctl_min": st.session_state.get("rng_pctl_min_optimization", 1),
                        "rng_pctl_max": st.session_state.get("rng_pctl_max_optimization", 5),
                        "natr_period_min": st.session_state.get("natr_period_min_optimization", 5),
                        "natr_period_max": st.session_state.get("natr_period_max_optimization", 20),
                        "natr_min_min": round(st.session_state.get("natr_min_min_optimization", 0.2), 2),
                        "natr_min_max": round(st.session_state.get("natr_min_max_optimization", 0.8), 2),
                        "lookback_period_min": st.session_state.get("lookback_period_min_optimization", 10),
                        "lookback_period_max": st.session_state.get("lookback_period_max_optimization", 30),
                        "min_growth_pct_min": round(st.session_state.get("min_growth_pct_min_optimization", 0.5), 2),
                        "min_growth_pct_max": round(st.session_state.get("min_growth_pct_max_optimization", 2.0), 2),
                        "stop_loss_pct_min": round(st.session_state.get("stop_loss_pct_min_optimization", 1.0), 2),
                        "stop_loss_pct_max": round(st.session_state.get("stop_loss_pct_max_optimization", 5.0), 2),
                        "take_profit_pct_min": round(st.session_state.get("take_profit_pct_min_optimization", 1.0), 2),
                        "take_profit_pct_max": round(st.session_state.get("take_profit_pct_max_optimization", 8.0), 2),
                    })

                    # Добавляем параметры климаксного выхода для оптимизации
                    use_climax_exit_option_opt = st.session_state.get("use_climax_exit_option", "Нет")
                    if use_climax_exit_option_opt == "Да":
                        profile_data["use_climax_exit"] = ("categorical", [True])
                        profile_data["climax_exit_window_min"] = st.session_state.get("climax_exit_window_min_optimization", 20)
                        profile_data["climax_exit_window_max"] = st.session_state.get("climax_exit_window_max_optimization", 100)
                        profile_data["climax_exit_threshold_min"] = st.session_state.get("climax_exit_threshold_min_optimization", 2)
                        profile_data["climax_exit_threshold_max"] = st.session_state.get("climax_exit_threshold_max_optimization", 5)

                    # Добавляем параметры "вилки" для оптимизации
                    profile_data["bracket_offset_pct_min_optimization"] = st.session_state.get("bracket_offset_pct_min_optimization", 0.1)
                    profile_data["bracket_offset_pct_max_optimization"] = st.session_state.get("bracket_offset_pct_max_optimization", 1.0)
                    profile_data["bracket_timeout_candles_min_optimization"] = st.session_state.get("bracket_timeout_candles_min_optimization", 2)
                    profile_data["bracket_timeout_candles_max_optimization"] = st.session_state.get("bracket_timeout_candles_max_optimization", 10)

                # Сохраняем список выбранных файлов для обоих модулей
                selected_files_key = f"selected_files_{module}"
                profile_data["selected_files"] = st.session_state.get(selected_files_key, [])
                profile_data["name"] = profile_name
                if save_profile(profile_name, profile_data, module):
                    # Очищаем кэш, чтобы список профилей сразу обновился
                    get_profiles.clear()
                    st.rerun()
            else:
                st.warning("Введите название профиля")
    with col2:
        profiles = get_profiles(module)
        selected_profile = st.selectbox(f"Загрузить {'профиль' if module == 'analysis' else 'диапазоны'}", options=profiles, key=f"select_profile_{module}")
        if st.button(f"Загрузить {'профиль' if module == 'analysis' else 'диапазоны'}", key=f"load_profile_{module}") and selected_profile:
            profile_data = load_profile(selected_profile, module)
            if profile_data:
                # Используем специальный флаг для обновления session_state без конфликта с виджетами
                st.session_state["profile_to_load"] = profile_data
                st.session_state["page_to_rerun"] = "Анализ сигналов" if module == "analysis" else "Оптимизация"
                st.rerun()

# Функция для получения списка файлов прогонов оптимизации
@st.cache_data(ttl=10)  # Кэшируем на 10 секунд
def get_optimization_run_files():
    """Получает список файлов прогонов оптимизации, отсортированных по дате."""
    try:
        files = [f for f in os.listdir("optimization_runs") if f.endswith('.json')]
        
        def extract_datetime_from_filename(f):
            try:
                # Извлекаем дату и время из имени файла формата run_YYYYMMDD_HHMMSS_...
                datetime_str = f.split('_')[1] + '_' + f.split('_')[2]
                return datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
            except (IndexError, ValueError):
                # Если формат имени файла другой, используем дату модификации файла
                try:
                    return datetime.fromtimestamp(os.path.getmtime(os.path.join("optimization_runs", f)))
                except OSError:
                    return datetime.min

        # Сортируем файлы от новых к старым
        sorted_files = sorted(files, key=extract_datetime_from_filename, reverse=True)
        return sorted_files
    except FileNotFoundError:
        return []

# Функция для загрузки данных прогона оптимизации
@st.cache_data(ttl=300)  # Кэшируем на 5 минут
def load_run_data_cached(run_file):
    """Загружает и кэширует данные одного прогона оптимизации из JSON-файла."""
    file_path = os.path.join("optimization_runs", run_file)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, OSError) as e:
        st.error(f"Ошибка чтения файла прогона '{run_file}': {str(e)}")
        return None
    except FileNotFoundError:
        st.error(f"Файл прогона '{run_file}' не найден")
        return None
    except json.JSONDecodeError:
        st.error(f"Файл прогона '{run_file}' содержит некорректный JSON")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке прогона '{run_file}': {str(e)}")
        return None