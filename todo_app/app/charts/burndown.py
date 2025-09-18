import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from ..services import task_service, category_service
from ..data import db


def render_burndown_chart():
    st.header("📈 バーンダウンチャート")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "開始日",
            value=date.today() - timedelta(days=30),
            key="burndown_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "終了日",
            value=date.today(),
            key="burndown_end_date"
        )
    
    if start_date >= end_date:
        st.error("開始日は終了日より前である必要があります。")
        return
    
    categories = category_service.get_all_categories()
    category_options = ["すべて"] + [cat.name for cat in categories]
    selected_category = st.selectbox("カテゴリでフィルタ", category_options)
    
    category_id = None
    if selected_category != "すべて":
        category_id = next((cat.id for cat in categories if cat.name == selected_category), None)
    
    chart_data = generate_burndown_data(start_date, end_date, category_id)
    
    if not chart_data:
        st.info("選択した期間にデータがありません。")
        return
    
    fig = create_burndown_chart(chart_data, start_date, end_date)
    st.plotly_chart(fig, use_container_width=True)
    
    render_sprint_summary(chart_data)


def generate_burndown_data(start_date: date, end_date: date, category_id: Optional[int]) -> List[Dict]:
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    query = """
    SELECT DATE(created_at) as date, COUNT(*) as created_count
    FROM tasks
    WHERE created_at BETWEEN ? AND ?
    """
    params = [start_datetime, end_datetime]
    
    if category_id:
        query += " AND category_id = ?"
        params.append(category_id)
    
    query += " GROUP BY DATE(created_at) ORDER BY date"
    
    created_data = db.execute_query(query, tuple(params))
    
    query = """
    SELECT DATE(completed_at) as date, COUNT(*) as completed_count
    FROM tasks
    WHERE completed_at BETWEEN ? AND ?
    AND status = 'done'
    """
    params = [start_datetime, end_datetime]
    
    if category_id:
        query += " AND category_id = ?"
        params.append(category_id)
    
    query += " GROUP BY DATE(completed_at) ORDER BY date"
    
    completed_data = db.execute_query(query, tuple(params))
    
    created_dict = {}
    for row in created_data:
        date_key = row['date']
        if isinstance(date_key, str):
            date_key = datetime.strptime(date_key, '%Y-%m-%d').date()
        elif isinstance(date_key, datetime):
            date_key = date_key.date()
        created_dict[date_key] = row['created_count']
    
    completed_dict = {}
    for row in completed_data:
        date_key = row['date']
        if isinstance(date_key, str):
            date_key = datetime.strptime(date_key, '%Y-%m-%d').date()
        elif isinstance(date_key, datetime):
            date_key = date_key.date()
        completed_dict[date_key] = row['completed_count']
    
    chart_data = []
    current_date = start_date
    total_tasks = 0
    remaining_tasks = 0
    
    total_at_start_query = """
    SELECT COUNT(*) as total
    FROM tasks
    WHERE created_at <= ?
    """
    start_params = [start_datetime]
    
    if category_id:
        total_at_start_query += " AND category_id = ?"
        start_params.append(category_id)
    
    total_at_start = db.execute_query_one(total_at_start_query, tuple(start_params))
    remaining_tasks = total_at_start['total'] if total_at_start else 0
    
    while current_date <= end_date:
        created_today = created_dict.get(current_date, 0)
        completed_today = completed_dict.get(current_date, 0)
        
        remaining_tasks += created_today - completed_today
        
        chart_data.append({
            'date': current_date,
            'remaining_tasks': max(0, remaining_tasks),
            'created_tasks': created_today,
            'completed_tasks': completed_today
        })
        
        current_date += timedelta(days=1)
    
    return chart_data


def create_burndown_chart(data: List[Dict], start_date: date, end_date: date) -> go.Figure:
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['remaining_tasks'],
        mode='lines+markers',
        name='残りタスク数',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    ideal_start = df['remaining_tasks'].iloc[0] if len(df) > 0 else 0
    ideal_end = 0
    ideal_line = [ideal_start - (ideal_start * i / (len(df) - 1)) for i in range(len(df))]
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=ideal_line,
        mode='lines',
        name='理想線',
        line=dict(color='blue', width=2, dash='dash'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title='バーンダウンチャート',
        xaxis_title='日付',
        yaxis_title='残りタスク数',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def render_sprint_summary(data: List[Dict]):
    if not data:
        return
    
    st.subheader("📊 期間サマリー")
    
    total_created = sum(day['created_tasks'] for day in data)
    total_completed = sum(day['completed_tasks'] for day in data)
    start_remaining = data[0]['remaining_tasks'] if data else 0
    end_remaining = data[-1]['remaining_tasks'] if data else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("作成タスク", total_created)
    
    with col2:
        st.metric("完了タスク", total_completed)
    
    with col3:
        st.metric("残りタスク", end_remaining)
    
    with col4:
        completion_rate = (total_completed / (total_created + start_remaining) * 100) if (total_created + start_remaining) > 0 else 0
        st.metric("完了率", f"{completion_rate:.1f}%")
    
    render_daily_progress_chart(data)


def render_daily_progress_chart(data: List[Dict]):
    st.subheader("📅 日別進捗")
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['created_tasks'],
        name='作成',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['completed_tasks'],
        name='完了',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='日別タスク作成・完了数',
        xaxis_title='日付',
        yaxis_title='タスク数',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)