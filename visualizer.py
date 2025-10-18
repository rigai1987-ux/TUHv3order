import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from signal_generator import generate_signals # Этот импорт используется ниже
import os

def save_figure_to_html(fig, filename="plot.html"):
    """
    Сохраняет объект фигуры Plotly в HTML-файл.

    Args:
        fig: объект фигуры Plotly.
        filename: имя файла для сохранения.

    Returns:
        Путь к сохраненному файлу или None в случае ошибки.
    """
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    fig.write_html(filepath)
    return filepath

def plot_wfo_results(wfo_summary_df, equity_curve_df, aggregated_metrics):
    """
    Визуализирует результаты Walk-Forward Optimization.

    Args:
        wfo_summary_df (pd.DataFrame): Сводная таблица по шагам WFO.
        equity_curve_df (pd.DataFrame): DataFrame с кривой доходности ('exit_time', 'cumulative_pnl').
        aggregated_metrics (dict): Итоговые метрики.

    Returns:
        plotly.graph_objects.Figure: Фигура с графиками.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Совокупная Out-of-Sample доходность (Итого PnL: ${aggregated_metrics.get('total_out_of_sample_pnl', 0):.2f})",
            "PnL по каждому шагу WFO"
        ),
        row_heights=[0.7, 0.3]
    )

    # 1. График кривой доходности
    if not equity_curve_df.empty:
        fig.add_trace(go.Scatter(
            x=equity_curve_df['exit_time'],
            y=equity_curve_df['cumulative_pnl'],
            mode='lines',
            name='Equity Curve',
            line=dict(color='royalblue', width=2)
        ), row=1, col=1)

    # 2. Гистограмма PnL по шагам
    colors = ['limegreen' if pnl >= 0 else 'crimson' for pnl in wfo_summary_df['out_sample_pnl']]
    fig.add_trace(go.Bar(
        x=wfo_summary_df['test_period'],
        y=wfo_summary_df['out_sample_pnl'],
        name='PnL за шаг',
        marker_color=colors
    ), row=2, col=1)

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=False,
        title_text="Анализ стабильности стратегии (Walk-Forward)",
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Совокупный PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    fig.update_xaxes(title_text="Период тестирования", row=2, col=1)

    return fig

def plot_wfo_parameter_stability(wfo_summary_df: pd.DataFrame, param_space: dict):
    """
    Визуализирует стабильность найденных параметров на протяжении шагов WFO.

    Args:
        wfo_summary_df (pd.DataFrame): Сводная таблица по шагам WFO.
        param_space (dict): Пространство параметров, чтобы определить, какие столбцы являются параметрами.

    Returns:
        plotly.graph_objects.Figure or None: Фигура с графиками или None, если нет данных.
    """
    # Отбираем только те столбцы, которые являются оптимизируемыми параметрами
    optimized_params = [p for p in param_space.keys() if p in wfo_summary_df.columns and wfo_summary_df[p].nunique() > 1]

    if not optimized_params:
        st.info("Не найдено изменяющихся параметров для построения графика стабильности.")
        return None

    # Определяем количество строк и столбцов для сетки графиков
    num_params = len(optimized_params)
    cols = min(3, num_params)
    rows = (num_params + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=optimized_params,
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )

    for i, param_name in enumerate(optimized_params):
        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(go.Scatter(
            x=wfo_summary_df['step'],
            y=wfo_summary_df[param_name],
            mode='lines+markers',
            name=param_name
        ), row=row, col=col)
        fig.update_yaxes(title_text=param_name, row=row, col=col)

    fig.update_layout(
        height=300 * rows,
        template='plotly_dark',
        showlegend=False,
        title_text="Стабильность оптимальных параметров между шагами WFO"
    )
    fig.update_xaxes(title_text="Шаг WFO")

    return fig

def plot_wfo_insample_vs_outsample(wfo_summary_df: pd.DataFrame):
    """
    Сравнивает метрику на обучающей выборке (In-Sample) с PnL на тестовой (Out-of-Sample).

    Args:
        wfo_summary_df (pd.DataFrame): Сводная таблица по шагам WFO.

    Returns:
        plotly.graph_objects.Figure or None: Фигура с графиком или None.
    """
    if 'in_sample_metric' not in wfo_summary_df.columns or 'out_sample_pnl' not in wfo_summary_df.columns:
        return None

    df = wfo_summary_df.copy()

    # Для многоцелевой оптимизации берем первую метрику
    if isinstance(df['in_sample_metric'].iloc[0], (list, tuple)):
        df['in_sample_metric_value'] = df['in_sample_metric'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
    else:
        df['in_sample_metric_value'] = df['in_sample_metric']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # График PnL (столбцы)
    colors = ['limegreen' if pnl >= 0 else 'crimson' for pnl in df['out_sample_pnl']]
    fig.add_trace(go.Bar(
        x=df['step'],
        y=df['out_sample_pnl'],
        name='Out-of-Sample PnL',
        marker_color=colors
    ), secondary_y=False)

    # График In-Sample метрики (линия)
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['in_sample_metric_value'],
        mode='lines+markers',
        name='In-Sample Metric (SQN/Score)',
        line=dict(color='cyan', width=2)
    ), secondary_y=True)

    fig.update_layout(height=500, template='plotly_dark', title_text="Сравнение In-Sample метрики и Out-of-Sample PnL")
    fig.update_xaxes(title_text="Шаг WFO")
    fig.update_yaxes(title_text="Out-of-Sample PnL ($)", secondary_y=False)
    fig.update_yaxes(title_text="In-Sample Metric", secondary_y=True)

    return fig

def plot_wfo_feature_importance(wfo_summary_df: pd.DataFrame):
    """
    Визуализирует стабильность важности признаков (Feature Importance) на протяжении шагов WFO.

    Args:
        wfo_summary_df (pd.DataFrame): Сводная таблица по шагам WFO, содержащая столбец 'feature_importances'.

    Returns:
        plotly.graph_objects.Figure or None: Фигура с графиком или None, если нет данных.
    """
    if 'feature_importances' not in wfo_summary_df.columns:
        return None

    # 1. Извлекаем и преобразуем данные
    records = []
    for _, row in wfo_summary_df.iterrows():
        importances = row['feature_importances']
        if isinstance(importances, dict):
            record = {'step': row['step']}
            record.update(importances)
            records.append(record)

    if not records:
        st.info("Нет данных о важности признаков для отображения.")
        return None

    importance_df = pd.DataFrame(records).set_index('step')
    # Убираем признаки, которые всегда были равны 0
    importance_df = importance_df.loc[:, (importance_df != 0).any(axis=0)]

    if importance_df.empty:
        st.info("Все признаки имели нулевую важность на всех шагах.")
        return None

    # 2. Создаем график
    fig = go.Figure()

    # Используем stacked bar chart для наглядности
    for feature_name in importance_df.columns:
        fig.add_trace(go.Bar(
            x=importance_df.index,
            y=importance_df[feature_name],
            name=feature_name
        ))

    fig.update_layout(
        barmode='stack',
        height=600, template='plotly_dark',
        title_text="Стабильность важности признаков (Feature Importance) по шагам WFO",
        xaxis_title="Шаг WFO", yaxis_title="Важность признака"
    )
    return fig

def plot_single_trade(df, trade, window_size=50, params=None):
    """
    Отображение одной сделки на графике с окружающим контекстом.

    Args:
        df (pd.DataFrame): DataFrame с полными рыночными данными.
        trade (dict): Словарь с информацией о сделке.
        window_size (int): Количество свечей для отображения до и после сделки.
        params (dict, optional): Параметры стратегии для отображения всех сигналов.

    Returns:
        plotly.graph_objects.Figure: Фигура с графиком сделки.
    """
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']

    # Определяем диапазон для отображения
    start_idx = max(0, entry_idx - window_size)
    end_idx = min(len(df), exit_idx + window_size)

    df_trade = df.iloc[start_idx:end_idx]

    fig = go.Figure()

    # Добавляем свечной график
    fig.add_trace(go.Candlestick(
        x=df_trade['datetime'],
        open=df_trade['open'],
        high=df_trade['high'],
        low=df_trade['low'],
        close=df_trade['close'],
        name='Цена',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Добавляем маркер входа
    entry_time = df.iloc[entry_idx]['datetime']
    entry_price = trade['entry_price']
    direction = trade['direction']
    entry_marker_symbol = 'triangle-up' if direction == 'long' else 'triangle-down'
    entry_marker_color = 'lime'

    # --- Подготовка hover-текста для маркера входа ---
    hover_template = f'Вход в {direction.upper()}<br>Цена: %{{y:.5f}}'
    ml_features = trade.get('ml_features')
    if ml_features:
        hover_template += '<br>--- ML Features ---'
        # Округляем значения для читаемости
        for feature, value in ml_features.items():
            if isinstance(value, float):
                hover_template += f'<br>{feature}: {value:.3f}'
            else:
                hover_template += f'<br>{feature}: {value}'
    hover_template += '<extra></extra>' # Убирает дополнительную информацию plotly
    # --- Конец подготовки hover-текста ---

    fig.add_trace(go.Scatter(
        x=[entry_time], y=[entry_price],
        mode='markers', name=f'Вход в {direction.upper()}',
        marker=dict(symbol=entry_marker_symbol, size=12, color='white', line=dict(width=2, color=entry_marker_color)),
        hovertemplate=hover_template
    ))

    # Добавляем маркер выхода
    exit_time = df.iloc[exit_idx]['datetime']
    exit_price = trade['exit_price']
    exit_reason = trade['exit_reason']
    fig.add_trace(go.Scatter(
        x=[exit_time], y=[exit_price],
        mode='markers', name=f'Выход ({exit_reason})',
        marker=dict(symbol='x', size=10, color='cyan', line=dict(width=2, color='darkcyan')),
        hovertemplate=f'Выход ({exit_reason})<br>Цена: %{{y:.5f}}<extra></extra>'
    ))

    # Добавляем все сигналы в этом окне
    if params:
        # Генерируем все базовые сигналы по текущим параметрам
        all_signal_indices, _, _ = generate_signals(df, params, base_signal_only=True)
        window_signals = {idx for idx in all_signal_indices if start_idx <= idx < end_idx}

        # Если это ML-симуляция, разделяем сигналы на одобренные и отклоненные
        is_ml_trade = 'ml_features' in trade and trade['ml_features'] is not None
        if is_ml_trade:
            # Сигналы, отклоненные моделью в этом окне
            rejected_indices = set(st.session_state.get("simulation_results", {}).get("ml_rejected_signals", []))
            rejected_in_window = {idx for idx in rejected_indices if start_idx <= idx < end_idx}
            
            # Одобренные сигналы (те, что не были отклонены)
            approved_in_window = window_signals - rejected_in_window

            if rejected_in_window:
                rejected_df = df.iloc[list(rejected_in_window)]
                fig.add_trace(go.Scatter(
                    x=rejected_df['datetime'], y=rejected_df['low'] * 0.998,
                    mode='markers', name='ML Rejected',
                    marker=dict(symbol='square', size=5, color='red', opacity=0.7),
                    hovertemplate='Сигнал отклонен ML<extra></extra>'
                ))
            
            if approved_in_window:
                approved_df = df.iloc[list(approved_in_window)]
                fig.add_trace(go.Scatter(
                    x=approved_df['datetime'], y=approved_df['low'] * 0.998,
                    mode='markers', name='ML Approved',
                    marker=dict(symbol='square', size=5, color='white', opacity=0.7),
                    hovertemplate='Сигнал одобрен ML<extra></extra>'
                ))
        elif window_signals: # Если не ML-симуляция, просто показываем все сигналы
            signals_df = df.iloc[list(window_signals)]
            fig.add_trace(go.Scatter(
                x=signals_df['datetime'], y=signals_df['low'] * 0.998,
                mode='markers', name='Все сигналы',
                marker=dict(symbol='square', size=5, color='white', opacity=0.7),
                hoverinfo='none'
            ))

    # --- Добавляем маркер на сигнальную свечу, которая привела к этой сделке ---
    signal_idx_for_trade = trade.get('signal_idx_for_trade')
    if signal_idx_for_trade is not None and start_idx <= signal_idx_for_trade < end_idx:
        signal_candle = df.iloc[signal_idx_for_trade]
        fig.add_trace(go.Scatter(
            x=[signal_candle['datetime']],
            y=[signal_candle['low'] * 0.995], # Чуть ниже остальных сигналов
            mode='markers',
            name='Сигнал для сделки',
            marker=dict(symbol='star', size=10, color='yellow'),
            hovertemplate=f'Сигнал, инициировавший сделку<br>Индекс: {signal_idx_for_trade}<extra></extra>'
        ))

    # --- Конец блока ---

    # Добавляем линии Stop Loss и Take Profit
    sl_price = trade.get('stop_loss')
    tp_price = trade.get('take_profit')
    
    if sl_price and tp_price:
        fig.add_hline(y=sl_price, line_dash="dash", line_color="orange", annotation_text="Stop Loss", annotation_position="bottom right")
        fig.add_hline(y=tp_price, line_dash="dash", line_color="lightgreen", annotation_text="Take Profit", annotation_position="bottom right")

    # Если это сделка по "вилке", отображаем уровни вилки
    # Логика "вилки" теперь единственная, поэтому отображаем ее уровни всегда, если есть параметры
    if params:
        # В новых версиях симулятора этот индекс должен сохраняться в сделке.
        signal_idx_for_trade = trade.get('signal_idx_for_trade', None)

        if signal_idx_for_trade is not None:
            # Убедимся, что индекс в пределах DataFrame и параметры доступны
            if signal_idx_for_trade < len(df) and params:
                base_price = df.iloc[signal_idx_for_trade]['close']
                offset_pct = params.get("bracket_offset_pct", 0.5)
                long_level = base_price * (1 + offset_pct / 100)
                short_level = base_price * (1 - offset_pct / 100)

                fig.add_hline(y=long_level, line_dash="dot", line_color="cyan", annotation_text="Long Level", annotation_position="top left")
                fig.add_hline(y=short_level, line_dash="dot", line_color="magenta", annotation_text="Short Level", annotation_position="bottom left")

    # Настройка макета
    pnl = trade.get('pnl', 0)
    title = f"Сделка #{trade.get('entry_idx')}: {direction.upper()} | PnL: ${pnl:.2f}"
    fig.update_layout(
        title=title,
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    return fig

def plot_ml_decision_boundary(df: pd.DataFrame, results: dict, params: dict):
    """
    Визуализирует, как ML-модель разделяет сигналы на основе двух самых важных признаков.

    Args:
        df (pd.DataFrame): Полный DataFrame с данными.
        results (dict): Результаты симуляции, содержащие 'feature_importances' и 'ml_rejected_signals'.
        params (dict): Параметры, использованные для симуляции.

    Returns:
        plotly.graph_objects.Figure or None: Фигура с графиком или None, если нет данных.
    """
    feature_importances = results.get('feature_importances')
    if not feature_importances:
        st.info("Нет данных о важности признаков для построения графика ML-анализа.")
        return None

    # 1. Находим два самых важных признака
    sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    
    # --- УЛУЧШЕНИЕ: Проверяем наличие хотя бы одного признака с ненулевой важностью ---
    # Отфильтровываем признаки с нулевой важностью
    sorted_features = [f for f in sorted_features if f[1] > 0]
    if not sorted_features:
        st.warning("Не найдено ни одного признака с важностью > 0 для построения графика.")
        return None

    # 2. Получаем DataFrame с признаками, который был использован при обучении/симуляции
    features_df = results.get('ml_features_df')
    if features_df is None or features_df.empty:
        st.info("DataFrame с признаками ML не найден в результатах. Невозможно построить график.")
        return None

    # 3. Готовим данные: разделяем сигналы на "одобренные" и "отклоненные"
    rejected_indices = set(results.get('ml_rejected_signals', []))
    
    # Создаем копию, чтобы не изменять исходный DataFrame в `results`
    plot_data_df = features_df.copy()
    plot_data_df['status'] = plot_data_df.index.map(lambda idx: 'Отклонен' if idx in rejected_indices else 'Одобрен')

    fig = go.Figure()

    # --- УЛУЧШЕНИЕ: Логика выбора типа графика в зависимости от количества ВАЖНЫХ признаков ---
    if len(sorted_features) >= 2:
        # --- Сценарий 1: Есть 2+ важных признака. Строим 2D-график (Scatter plot) ---
        feature1_name, _ = sorted_features[0]
        feature2_name, _ = sorted_features[1]

        if feature1_name not in plot_data_df.columns or feature2_name not in plot_data_df.columns:
            st.error(f"Ключевые признаки ('{feature1_name}' или '{feature2_name}') отсутствуют в 'ml_features_df'.")
            return None

        for status, color in [('Одобрен', 'lime'), ('Отклонен', 'red')]:
            subset_df = plot_data_df[plot_data_df['status'] == status]
            if subset_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset_df[feature1_name],
                y=subset_df[feature2_name],
                mode='markers',
                name=status,
                marker=dict(color=color, size=7, opacity=0.8, line=dict(width=1, color='Black')),
                hovertemplate=(
                    f"<b>{status}</b><br>"
                    f"{feature1_name}: %{{x}}<br>"
                    f"{feature2_name}: %{{y}}<br>"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=f"Разделение сигналов моделью по признакам '{feature1_name}' и '{feature2_name}'",
            xaxis_title=f"Признак: {feature1_name}",
            yaxis_title=f"Признак: {feature2_name}",
            legend_title_text='Статус сигнала',
            height=600,
            template='plotly_dark'
        )
    else:
        # --- Сценарий 2: Есть только 1 важный признак. Строим 1D-график (гистограмму) ---
        feature1_name, _ = sorted_features[0]
        if feature1_name not in plot_data_df.columns:
            st.error(f"Ключевой признак '{feature1_name}' отсутствует в 'ml_features_df'.")
            return None

        for status, color in [('Одобрен', 'lime'), ('Отклонен', 'red')]:
            subset_df = plot_data_df[plot_data_df['status'] == status]
            if subset_df.empty:
                continue
            fig.add_trace(go.Histogram(
                x=subset_df[feature1_name],
                name=status,
                marker_color=color,
                opacity=0.7,
                xbins=dict(size=(plot_data_df[feature1_name].max() - plot_data_df[feature1_name].min()) / 30) # Авто-размер корзин
            ))

        fig.update_layout(
            barmode='overlay', # Наложение гистограмм для сравнения
            title=f"Распределение сигналов по самому важному признаку: '{feature1_name}'",
            xaxis_title=f"Значение признака: {feature1_name}",
            yaxis_title="Количество сигналов",
            legend_title_text='Статус сигнала',
            height=500,
            template='plotly_dark'
        )

    return fig