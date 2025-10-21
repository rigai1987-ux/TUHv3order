import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from signal_generator import generate_signals # Этот импорт используется ниже

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
            f"Совокупная Out-of-Sample доходность (Итого PnL: ${aggregated_metrics.get('total_out_of_sample_pnl', 0):.2f}, "
            f"Max DD: {aggregated_metrics.get('overall_max_drawdown', 0):.2%}, "
            f"Sharpe: {aggregated_metrics.get('overall_sharpe_ratio', 0):.2f})",
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

def plot_wfo_comparison(results_ml: dict, results_no_ml: dict):
    """
    Сравнивает кривые доходности двух WFO-прогонов (с ML и без).

    Args:
        results_ml (dict): Результаты WFO-прогона с ML-фильтром.
        results_no_ml (dict): Результаты WFO-прогона без ML-фильтра.

    Returns:
        plotly.graph_objects.Figure: Фигура с графиком сравнения.
    """
    fig = go.Figure()

    # 1. Добавляем кривую доходности для прогона с ML
    equity_ml = results_ml.get('equity_curve')
    if equity_ml is not None and not equity_ml.empty:
        pnl_ml = results_ml['aggregated_metrics'].get('total_out_of_sample_pnl', 0)
        fig.add_trace(go.Scatter(
            x=equity_ml['exit_time'],
            y=equity_ml['cumulative_pnl'],
            mode='lines',
            name=f'С ML-фильтром (Итого PnL: ${pnl_ml:.2f})',
            line=dict(color='cyan', width=2)
        ))

    # 2. Добавляем кривую доходности для прогона без ML (Baseline)
    equity_no_ml = results_no_ml.get('equity_curve')
    if equity_no_ml is not None and not equity_no_ml.empty:
        pnl_no_ml = results_no_ml['aggregated_metrics'].get('total_out_of_sample_pnl', 0)
        fig.add_trace(go.Scatter(
            x=equity_no_ml['exit_time'],
            y=equity_no_ml['cumulative_pnl'],
            mode='lines',
            name=f'Без ML (Baseline) (Итого PnL: ${pnl_no_ml:.2f})',
            line=dict(color='orange', width=2, dash='dash')
        ))

    fig.update_layout(
        title="Сравнение кривых доходности WFO: ML-фильтр vs. Baseline",
        xaxis_title="Дата", yaxis_title="Совокупный PnL ($)",
        template='plotly_dark', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_wfo_ml_effectiveness(summary_ml_df: pd.DataFrame, summary_no_ml_df: pd.DataFrame = None):
    """
    Визуализирует эффективность ML-фильтра в WFO.

    1. Если передан только `summary_ml_df`, строит гистограмму отфильтрованных сделок.
    2. Если переданы оба DataFrame, строит сравнение Win Rate по шагам.

    Args:
        summary_ml_df (pd.DataFrame): Сводка WFO-прогона с ML.
        summary_no_ml_df (pd.DataFrame, optional): Сводка WFO-прогона без ML.

    Returns:
        plotly.graph_objects.Figure or None: Фигура с графиком или None.
    """
    if summary_ml_df.empty:
        return None

    # Если есть данные для сравнения, строим график сравнения Win Rate
    if summary_no_ml_df is not None and not summary_no_ml_df.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "Сравнение Win Rate по шагам WFO",
                "Эффективность ML-фильтра (отфильтровано сделок)"
            ),
            row_heights=[0.6, 0.4]
        )

        # График 1: Сравнение Win Rate
        fig.add_trace(go.Scatter(
            x=summary_ml_df['step'], y=summary_ml_df['out_sample_win_rate'],
            mode='lines+markers', name='Win Rate с ML', line=dict(color='cyan')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=summary_no_ml_df['step'], y=summary_no_ml_df['out_sample_win_rate'],
            mode='lines+markers', name='Win Rate без ML (Baseline)', line=dict(color='orange', dash='dash')
        ), row=1, col=1)

        fig.update_yaxes(title_text="Win Rate", tickformat=".0%", row=1, col=1)

        # График 2: Отфильтрованные сделки
        # Рассчитываем количество отфильтрованных сделок
        filtered_trades = summary_no_ml_df['out_sample_trades'] - summary_ml_df['out_sample_trades']
        fig.add_trace(go.Bar(
            x=summary_ml_df['step'], y=filtered_trades,
            name='Отфильтровано сделок', marker_color='lightslategrey'
        ), row=2, col=1)
        fig.update_yaxes(title_text="Кол-во сделок", row=2, col=1)
        fig.update_xaxes(title_text="Шаг WFO", row=2, col=1)

        fig.update_layout(height=700, template='plotly_dark', title_text="Анализ эффективности ML-фильтра в WFO")

    # Если данных для сравнения нет, строим только гистограмму отфильтрованных сделок
    else:
        # Для этого нам нужно знать, сколько было бы сделок без фильтра.
        # Эта информация есть в user_attrs, но не в итоговом summary.
        # Поэтому в этом режиме мы не можем построить этот график.
        # Вместо этого, можно показать, на каких шагах ML вообще применялся.
        fig = go.Figure()
        colors = ['limegreen' if applied else 'crimson' for applied in summary_ml_df['ml_applied']]
        fig.add_trace(go.Bar(
            x=summary_ml_df['step'], y=summary_ml_df['ml_applied'].astype(int),
            name='ML применялся', marker_color=colors
        ))
        fig.update_layout(height=400, template='plotly_dark', title_text="Применение ML-модели на шагах WFO", yaxis_title="ML применялся (1=Да, 0=Нет)", xaxis_title="Шаг WFO")

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
    hover_template = f'Вход в {direction.upper()}<br>Цена: %{{y:.5f}}<extra></extra>'
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
        all_signal_indices, _, _ = generate_signals(df, params)
        window_signals = {idx for idx in all_signal_indices if start_idx <= idx < end_idx}

        if window_signals:
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
    # --- ИЗМЕНЕНИЕ: Для пропущенных сделок линии делаем пунктирными ---
    line_style = "dot" if trade.get('skipped_by_ml') else "dash"
    sl_color = "rgba(255, 165, 0, 0.6)" if trade.get('skipped_by_ml') else "orange"
    tp_color = "rgba(144, 238, 144, 0.6)" if trade.get('skipped_by_ml') else "lightgreen"

    fig.add_hline(y=trade['stop_loss'], line_dash=line_style, line_color=sl_color, annotation_text="Stop Loss", annotation_position="bottom right")
    fig.add_hline(y=trade['take_profit'], line_dash=line_style, line_color=tp_color, annotation_text="Take Profit", annotation_position="bottom right")


    # Если это сделка по "вилке", отображаем уровни вилки
    # Логика "вилки" теперь единственная, поэтому отображаем ее уровни всегда, если есть параметры
    if params:
        # В новых версиях симулятора этот индекс должен сохраняться в сделке.
        signal_idx_for_trade = trade.get('signal_idx_for_trade', None)

        if signal_idx_for_trade is not None:
            # Убедимся, что индекс в пределах DataFrame и параметры доступны
            if signal_idx_for_trade < len(df) - 1 and params:
                # --- ИСПРАВЛЕНИЕ: Используем цену открытия СЛЕДУЮЩЕЙ свечи, как в симуляторе ---
                base_price = df.iloc[signal_idx_for_trade + 1]['open']
                offset_pct = params.get("bracket_offset_pct", 0.5)
                long_level = base_price * (1 + offset_pct / 100)
                short_level = base_price * (1 - offset_pct / 100)

                fig.add_hline(y=long_level, line_dash="dot", line_color="cyan", annotation_text="Long Level", annotation_position="top left")
                fig.add_hline(y=short_level, line_dash="dot", line_color="magenta", annotation_text="Short Level", annotation_position="bottom left")

    # Настройка макета
    pnl = trade.get('pnl', 0)
    title_font_color = 'white' # Стандартный цвет для темной темы
    # --- ИЗМЕНЕНИЕ: Добавляем пометку для пропущенных сделок ---
    if trade.get('skipped_by_ml'):
        title = f"ПРОПУЩЕНО ML: {direction.upper()} | Сигнал #{trade.get('signal_idx_for_trade')}"
        title_font_color = 'crimson' # Выделяем красным
    else:
        title = f"Сделка #{trade.get('entry_idx')}: {direction.upper()} | PnL: ${pnl:.2f}"
    fig.update_layout(
        title=title,
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        title_font_color=title_font_color
    )

    return fig

def plot_wfo_risk_metrics(wfo_summary_df: pd.DataFrame):
    """
    Визуализирует метрики риска и доходности по шагам WFO.

    Args:
        wfo_summary_df (pd.DataFrame): Сводная таблица по шагам WFO.

    Returns:
        plotly.graph_objects.Figure or None: Фигура с графиками или None, если нет данных.
    """
    if wfo_summary_df.empty:
        return None

    # Проверяем наличие необходимых колонок
    required_cols = ['out_sample_max_drawdown', 'out_sample_sharpe_ratio', 'out_sample_profit_factor']
    if not all(col in wfo_summary_df.columns for col in required_cols):
        st.warning("Недостаточно данных для построения графиков метрик риска (отсутствуют Max Drawdown, Sharpe Ratio или Profit Factor).")
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Максимальная просадка (Out-of-Sample) по шагам WFO",
            "Коэффициент Шарпа (Out-of-Sample) по шагам WFO",
            "Профит-фактор (Out-of-Sample) по шагам WFO"
        )
    )

    # 1. Максимальная просадка
    fig.add_trace(go.Bar(
        x=wfo_summary_df['test_period'],
        y=wfo_summary_df['out_sample_max_drawdown'],
        name='Max Drawdown',
        marker_color='orange'
    ), row=1, col=1)
    fig.update_yaxes(title_text="Max Drawdown (%)", tickformat=".2%", row=1, col=1)

    # 2. Коэффициент Шарпа
    colors_sharpe = ['limegreen' if sr >= 0 else 'crimson' for sr in wfo_summary_df['out_sample_sharpe_ratio']]
    fig.add_trace(go.Bar(
        x=wfo_summary_df['test_period'],
        y=wfo_summary_df['out_sample_sharpe_ratio'],
        name='Sharpe Ratio',
        marker_color=colors_sharpe
    ), row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)

    # 3. Профит-фактор
    colors_pf = ['limegreen' if pf >= 1 else 'crimson' for pf in wfo_summary_df['out_sample_profit_factor']]
    fig.add_trace(go.Bar(
        x=wfo_summary_df['test_period'],
        y=wfo_summary_df['out_sample_profit_factor'],
        name='Profit Factor',
        marker_color=colors_pf
    ), row=3, col=1)
    fig.update_yaxes(title_text="Profit Factor", row=3, col=1)
    fig.update_xaxes(title_text="Период тестирования", row=3, col=1)

    fig.update_layout(
        height=900,
        template='plotly_dark',
        showlegend=False,
        title_text="Анализ метрик риска и доходности по шагам WFO"
    )

    return fig

def plot_ml_decision_boundary(model_bundle, X, y, feature1, feature2):
    """
    Визуализирует границу принятия решений для ML-модели на двух самых важных признаках.

    Args:
        model_bundle (dict): Словарь с обученной моделью, скейлером и т.д.
        X (pd.DataFrame): DataFrame с признаками, использованными для обучения.
        y (pd.Series): Series с реальными метками.
        feature1 (str): Имя признака для оси X.
        feature2 (str): Имя признака для оси Y.

    Returns:
        plotly.graph_objects.Figure: Фигура с графиком.
    """
    # 1. Извлекаем компоненты из бандла
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    numerical_features = model_bundle['numerical_features']
    categorical_features = model_bundle.get('categorical_features', []) # Получаем список категориальных признаков
    all_feature_names = model_bundle['feature_names']

    # 3. Подготавливаем данные для графика (только 2 признака)
    X_plot = X[[feature1, feature2]].copy()
    
    # 4. Создаем сетку (mesh grid) для построения фона
    x_min, x_max = X_plot[feature1].min(), X_plot[feature1].max()
    y_min, y_max = X_plot[feature2].min(), X_plot[feature2].max()
    
    # Добавляем отступы для красоты
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    
    xx, yy = np.meshgrid(np.linspace(x_min - x_pad, x_max + x_pad, 100),
                         np.linspace(y_min - y_pad, y_max + y_pad, 100))

    # 5. Делаем предсказания для каждой точки сетки
    # Создаем DataFrame для предсказания, который имеет все нужные модели признаки
    grid_points_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature1, feature2])
    
    # Добавляем остальные признаки, заполняя их средними (для числовых) или модой (для категориальных)
    for col in all_feature_names:
        if col not in grid_points_df.columns:
            if col in categorical_features:
                # Для категориальных используем моду (самое частое значение)
                grid_points_df[col] = X[col].mode()[0]
            else:
                # Для числовых используем среднее
                grid_points_df[col] = X[col].mean()

    # --- ИСПРАВЛЕНИЕ: Корректно масштабируем признаки, сохраняя типы категориальных ---
    # Создаем копию DataFrame для предсказания
    features_for_prediction = grid_points_df.copy()

    # 1. Масштабируем только числовые признаки
    # Убедимся, что scaler не пустой и есть числовые признаки для масштабирования
    if scaler and numerical_features:
        # Создаем список числовых признаков, которые действительно присутствуют в DataFrame
        num_features_in_df = [f for f in numerical_features if f in features_for_prediction.columns]
        if num_features_in_df:
            features_for_prediction[num_features_in_df] = scaler.transform(features_for_prediction[num_features_in_df])

    # 2. Приводим категориальные признаки к целочисленному типу, как ожидает модель
    for col in categorical_features:
        features_for_prediction[col] = features_for_prediction[col].astype(int)

    # Предсказываем класс для каждой точки сетки
    Z = model.predict(features_for_prediction[all_feature_names])
    Z = Z.reshape(xx.shape)

    # 6. Создаем график
    fig = go.Figure()

    # Добавляем фон (границы решений)
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(0, 255, 0, 0.2)']],
        showscale=False,
        line_width=0,
        name='Decision Boundary'
    ))

    # Добавляем точки реальных данных
    # Разделяем на успешные и провальные для легенды
    success_signals = X_plot[y == 1]
    fail_signals = X_plot[y == 0]

    fig.add_trace(go.Scatter(
        x=fail_signals[feature1], y=fail_signals[feature2],
        mode='markers', name='Провальные сигналы (0)',
        marker=dict(color='red', symbol='x', size=5, opacity=0.7)
    ))
    fig.add_trace(go.Scatter(
        x=success_signals[feature1], y=success_signals[feature2],
        mode='markers', name='Успешные сигналы (1)',
        marker=dict(color='lime', symbol='circle', size=5, opacity=0.7)
    ))

    # Настройка макета
    fig.update_layout(
        title=f"Границы решений модели по признакам '{feature1}' и '{feature2}'",
        xaxis_title=feature1,
        yaxis_title=feature2,
        template='plotly_dark',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_wfo_report_html(
    results_ml: dict,
    results_no_ml: dict,
    param_space: dict,
    run_name: str = "WFO Report"
) -> str:
    """
    Создает единый HTML-файл с полными результатами WFO-сравнения.

    Args:
        results_ml (dict): Результаты WFO-прогона с ML-фильтром.
        results_no_ml (dict): Результаты WFO-прогона без ML-фильтра.
        param_space (dict): Пространство параметров, использованное для оптимизации.
        run_name (str): Название прогона для заголовка отчета.

    Returns:
        str: Строка, содержащая полный HTML-код отчета.
    """
    # --- Подготовка данных и графиков ---
    summary_ml_df = pd.DataFrame(results_ml.get('summary', []))
    summary_no_ml_df = pd.DataFrame(results_no_ml.get('summary', []))

    # Сравнительная таблица метрик
    comparison_metrics_df = pd.DataFrame([
        {"Метод": "С ML-фильтром", **results_ml['aggregated_metrics']},
        {"Метод": "Без ML (Baseline)", **results_no_ml['aggregated_metrics']}
    ]).set_index("Метод").T # Транспонируем для лучшей читаемости

    # Генерация всех необходимых графиков
    fig_comparison_equity = plot_wfo_comparison(results_ml, results_no_ml)
    fig_ml_effectiveness = plot_wfo_ml_effectiveness(summary_ml_df, summary_no_ml_df)
    fig_param_stability_ml = plot_wfo_parameter_stability(summary_ml_df, param_space)
    fig_param_stability_no_ml = plot_wfo_parameter_stability(summary_no_ml_df, param_space)
    fig_risk_ml = plot_wfo_risk_metrics(summary_ml_df)
    fig_risk_no_ml = plot_wfo_risk_metrics(summary_no_ml_df)

    # Конвертация графиков в HTML (без полной обертки и без повторного включения JS)
    html_comparison_equity = fig_comparison_equity.to_html(full_html=False, include_plotlyjs=False) if fig_comparison_equity else ""
    html_ml_effectiveness = fig_ml_effectiveness.to_html(full_html=False, include_plotlyjs=False) if fig_ml_effectiveness else ""
    html_param_stability_ml = fig_param_stability_ml.to_html(full_html=False, include_plotlyjs=False) if fig_param_stability_ml else ""
    html_param_stability_no_ml = fig_param_stability_no_ml.to_html(full_html=False, include_plotlyjs=False) if fig_param_stability_no_ml else ""
    html_risk_ml = fig_risk_ml.to_html(full_html=False, include_plotlyjs=False) if fig_risk_ml else ""
    html_risk_no_ml = fig_risk_no_ml.to_html(full_html=False, include_plotlyjs=False) if fig_risk_no_ml else ""

    # --- Сборка HTML-документа ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{run_name}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.6;
                color: #e6e6e6;
                background-color: #0e1117;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: auto;
            }}
            h1, h2, h3 {{
                color: #fafafa;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #1a1f2b;
            }}
            tr:nth-child(even) {{
                background-color: #161a24;
            }}
            .plotly-graph-div {{
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Отчет по Walk-Forward Optimization: {run_name}</h1>
            <p><strong>Дата генерации:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Итоговые метрики</h2>
            {comparison_metrics_df.to_html(classes='table table-striped', float_format='{:.2f}'.format)}

            <h2>Сравнение кривых доходности</h2>
            {html_comparison_equity}

            <h2>Анализ эффективности ML-фильтра</h2>
            {html_ml_effectiveness}

            <hr>

            <h2>Детализация: Прогон с ML-фильтром</h2>
            <h3>Метрики риска и доходности по шагам (с ML)</h3>
            {html_risk_ml}
            <h3>Стабильность параметров (с ML)</h3>
            {html_param_stability_ml}
            <h3>Сводная таблица по шагам (с ML)</h3>
            {summary_ml_df.to_html(classes='table table-striped', float_format='{:.2f}'.format, index=False)}

            <hr>

            <h2>Детализация: Прогон без ML (Baseline)</h2>
            <h3>Метрики риска и доходности по шагам (без ML)</h3>
            {html_risk_no_ml}
            <h3>Стабильность параметров (без ML)</h3>
            {html_param_stability_no_ml}
            <h3>Сводная таблица по шагам (без ML)</h3>
            {summary_no_ml_df.to_html(classes='table table-striped', float_format='{:.2f}'.format, index=False)}

        </div>
    </body>
    </html>
    """
    return html_template