import streamlit as st
import pandas as pd
import json
import os
import datetime
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
                    date_obj = datetime.datetime.strptime(value.split()[0], "%Y-%m-%d").date()
                except ValueError:
                    st.warning(f"Не удалось распознать формат даты для '{key}' (значение: '{value}'). Пропускаем.", icon="⚠️")
                    continue # Пропускаем этот ключ, чтобы не записать некорректную строку
                st.session_state[f"{key}_{module}"] = date_obj
            # Обработка категориальных параметров для оптимизации
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
    params = {}
    suffix = f"_{module}"

    # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Унифицированный сбор всех параметров напрямую из session_state ---
    # Это гарантирует, что параметры будут собраны, даже если виджеты для них
    # отключены (disabled=True).
    # Собираем все ключи из session_state, которые заканчиваются на суффикс модуля.
    for session_key in st.session_state:
        if session_key.endswith(suffix):
            # Убираем суффикс, чтобы получить "чистый" ключ параметра
            clean_key = session_key[:-len(suffix)]
            # Пропускаем ключи, связанные с управлением UI, а не параметрами стратегии
            if not any(ui_part in session_key for ui_part in ['profile_name', 'select_profile', 'select_all_data', 'csv_', 'run_analysis_and_plot', 'select_all_data_prev']):
                params[clean_key] = st.session_state.get(session_key)

    # --- НОВОЕ: Принудительно включаем использование HLdir ---
    params['use_hldir_on_conflict'] = True

    # Удаляем ключи со значением None, чтобы не передавать лишнего в симулятор
    return {k: v for k, v in params.items() if v is not None}

# Функция для генерации параметров оптимизации (теперь использует универсальную функцию)
def get_optimization_parameters():
    """Получить параметры оптимизации (диапазоны) для модуля оптимизации"""
    # Эта функция теперь просто собирает диапазоны из UI, которые определены в optimization_page.py
    # Для сохранения совместимости с manage_profiles, она может остаться как заглушка
    # или быть переработана для сбора данных из session_state, как get_strategy_parameters.
    return {} # Основная логика сбора диапазонов находится в get_param_space_from_ui

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
            st.session_state[f"start_date_{module}"] = datetime.date(2025, 9, 1)
        # Создаем виджет с явным указанием значения из session_state
        start_date = st.date_input("Дата начала", value=st.session_state[f"start_date_{module}"], key=f"start_date_{module}")
    with col2:
        # Инициализируем значение в session_state, если его нет
        if f"end_date_{module}" not in st.session_state:
            st.session_state[f"end_date_{module}"] = datetime.date(2025, 9, 30)
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
        profile_name = st.text_input("Название профиля", key=f"profile_name_{module}", value="")
        if st.button(f"Сохранить {'профиль' if module == 'analysis' else 'диапазоны'}", key=f"save_profile_{module}"):
            if profile_name:
                # Сбор всех параметров стратегии
                profile_data = {
                    "position_size": st.session_state.get(f"position_size_{module}", 1000.0),
                    "commission": st.session_state.get(f"commission_{module}", 0.1),
                    "entry_logic_mode": "Вилка отложенных ордеров",
                    "aggressive_mode": st.session_state.get(f"aggressive_mode_{module}", False),
                    "start_date": str(st.session_state.get(f"start_date_{module}", datetime.date(2025, 9, 1))),
                    "end_date": str(st.session_state.get(f"end_date_{module}", datetime.date(2025, 9, 30)))
                }
                if module == "analysis":
                    # При сохранении профиля анализа, получаем параметры из session_state напрямую
                    # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Используем улучшенную функцию для сбора ВСЕХ параметров ---
                    # get_strategy_parameters соберет все параметры из session_state, включая те,
                    # что были загружены из аналитики и не имеют активных виджетов.
                    all_strategy_params = get_strategy_parameters("analysis")
                    profile_data.update(all_strategy_params)
                elif module == "optimization":
                    # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ: Автоматический сбор ВСЕХ параметров оптимизации ---
                    # Вместо ручного перечисления каждого параметра, мы итерируемся по session_state
                    # и собираем все ключи, которые относятся к оптимизации.
                    # Это делает код гибким и устойчивым к добавлению новых параметров в UI.
                    suffix = "_optimization"
                    for session_key, value in st.session_state.items():
                        if session_key.endswith(suffix):
                            # Пропускаем ключи, связанные с управлением UI (кнопки, выбор профиля и т.д.)
                            if not any(ui_part in session_key for ui_part in ['profile_name', 'select_profile', 'select_all_data', 'csv_', 'save_profile', 'load_profile']):
                                # Убираем суффикс, чтобы получить "чистый" ключ
                                clean_key = session_key[:-len(suffix)]
                                # --- ИСПРАВЛЕНИЕ: Преобразуем объекты date в строку перед сохранением ---
                                if isinstance(value, datetime.date):
                                    profile_data[clean_key] = str(value)
                                else:
                                    profile_data[clean_key] = value

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
                return datetime.datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
            except (IndexError, ValueError):
                # Если формат имени файла другой, используем дату модификации файла
                try:
                    return datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join("optimization_runs", f)))
                except OSError:
                    return datetime.datetime.min

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