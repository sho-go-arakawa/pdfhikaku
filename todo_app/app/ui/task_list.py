import streamlit as st
from datetime import datetime
from typing import Optional
from ..services import task_service, category_service
from ..models import TaskUpdate
from ..settings import PRIORITY_LEVELS, TASK_STATUSES


def render_task_filters():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.selectbox(
            "ステータス",
            options=["すべて"] + TASK_STATUSES,
            key="status_filter"
        )
    
    with col2:
        categories = category_service.get_all_categories()
        category_options = ["すべて"] + [cat.name for cat in categories]
        category_filter = st.selectbox(
            "カテゴリ",
            options=category_options,
            key="category_filter"
        )
    
    with col3:
        priority_options = ["すべて"] + [PRIORITY_LEVELS[i] for i in range(3)]
        priority_filter = st.selectbox(
            "優先度",
            options=priority_options,
            key="priority_filter"
        )
    
    with col4:
        search_term = st.text_input("検索", key="search_term")
    
    return {
        'status': None if status_filter == "すべて" else status_filter,
        'category_id': None if category_filter == "すべて" else next(
            (cat.id for cat in categories if cat.name == category_filter), None
        ),
        'priority': None if priority_filter == "すべて" else next(
            (i for i, label in PRIORITY_LEVELS.items() if label == priority_filter), None
        ),
        'search': search_term if search_term else None
    }


def render_task_list():
    st.header("📋 タスク一覧")
    
    filters = render_task_filters()
    
    if filters['search']:
        tasks = task_service.search_tasks(filters['search'])
    else:
        tasks = task_service.get_all_tasks(
            status=filters['status'],
            category_id=filters['category_id']
        )
    
    if filters['priority'] is not None:
        tasks = [task for task in tasks if task.priority == filters['priority']]
    
    if not tasks:
        st.info("該当するタスクがありません。")
        return
    
    for task in tasks:
        render_task_card(task)


def render_task_card(task):
    with st.container():
        col1, col2, col3, col4 = st.columns([0.1, 0.5, 0.2, 0.2])
        
        with col1:
            task_done = st.checkbox(
                "完了",
                value=task.status == "done",
                key=f"task_done_{task.id}",
                help="完了切替",
                label_visibility="collapsed"
            )
            
            if task_done != (task.status == "done"):
                task_service.toggle_task_status(task.id)
                st.rerun()
        
        with col2:
            priority_icon = get_priority_icon(task.priority)
            status_badge = get_status_badge(task.status)
            
            st.markdown(f"### {priority_icon} {task.title}")
            
            if task.description:
                st.markdown(f"*{task.description}*")
            
            info_items = []
            if task.category_name:
                info_items.append(f"📁 {task.category_name}")
            if task.due_date:
                due_color = "🔴" if task.due_date < datetime.utcnow() and task.status != "done" else "📅"
                info_items.append(f"{due_color} 期日: {task.due_date.strftime('%Y-%m-%d %H:%M')}")
            if task.tags:
                info_items.append(f"🏷️ {', '.join(task.tags)}")
            
            if info_items:
                st.markdown(" | ".join(info_items))
        
        with col3:
            st.markdown(status_badge)
            if task.completed_at and task.status == "done":
                st.markdown(f"*完了: {task.completed_at.strftime('%m/%d %H:%M')}*")
        
        with col4:
            edit_col, delete_col = st.columns(2)
            
            with edit_col:
                if st.button("✏️", key=f"edit_{task.id}", help="編集"):
                    st.session_state.edit_task_id = task.id
                    st.session_state.show_task_form = True
                    st.rerun()
            
            with delete_col:
                if st.button("🗑️", key=f"delete_{task.id}", help="削除"):
                    if st.session_state.get(f"confirm_delete_{task.id}", False):
                        task_service.delete_task(task.id)
                        st.success("タスクを削除しました")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{task.id}"] = True
                        st.warning("もう一度クリックして削除を確認してください")
        
        st.divider()


def get_priority_icon(priority: int) -> str:
    icons = {0: "🟢", 1: "🟡", 2: "🔴"}
    return icons.get(priority, "⚪")


def get_status_badge(status: str) -> str:
    badges = {
        "todo": "⏳ 未着手",
        "doing": "🔄 進行中", 
        "done": "✅ 完了"
    }
    return badges.get(status, status)