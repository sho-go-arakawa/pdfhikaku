import streamlit as st
from datetime import datetime, date, time
from typing import Optional
from ..services import task_service, category_service, tag_service
from ..models import TaskCreate, TaskUpdate
from ..settings import PRIORITY_LEVELS, RECURRENCE_RULES


def render_task_form():
    task_id = st.session_state.get('edit_task_id')
    is_edit = task_id is not None
    
    if is_edit:
        st.header("✏️ タスク編集")
        task = task_service.get_task_by_id(task_id)
        if not task:
            st.error("タスクが見つかりません")
            return
    else:
        st.header("➕ 新規タスク作成")
        task = None
    
    with st.form("task_form"):
        title = st.text_input(
            "タイトル *",
            value=task.title if task else "",
            max_chars=200
        )
        
        description = st.text_area(
            "詳細",
            value=task.description if task else "",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            priority = st.selectbox(
                "優先度",
                options=list(PRIORITY_LEVELS.keys()),
                format_func=lambda x: PRIORITY_LEVELS[x],
                index=task.priority if task else 1
            )
        
        with col2:
            status = st.selectbox(
                "ステータス",
                options=["todo", "doing", "done"],
                format_func=lambda x: {"todo": "未着手", "doing": "進行中", "done": "完了"}[x],
                index=["todo", "doing", "done"].index(task.status) if task else 0
            )
        
        categories = category_service.get_all_categories()
        category_options = [None] + categories
        category_labels = ["選択なし"] + [cat.name for cat in categories]
        
        selected_category_idx = 0
        if task and task.category_id:
            for i, cat in enumerate(categories):
                if cat.id == task.category_id:
                    selected_category_idx = i + 1
                    break
        
        category_idx = st.selectbox(
            "カテゴリ",
            options=range(len(category_options)),
            format_func=lambda x: category_labels[x],
            index=selected_category_idx
        )
        category_id = category_options[category_idx].id if category_options[category_idx] else None
        
        due_date_col, due_time_col = st.columns(2)
        
        with due_date_col:
            due_date_input = st.date_input(
                "期日",
                value=task.due_date.date() if task and task.due_date else None
            )
        
        with due_time_col:
            due_time_input = st.time_input(
                "期日時刻",
                value=task.due_date.time() if task and task.due_date else time(9, 0)
            )
        
        deadline_date_col, deadline_time_col = st.columns(2)
        
        with deadline_date_col:
            deadline_date_input = st.date_input(
                "締切",
                value=task.deadline.date() if task and task.deadline else None
            )
        
        with deadline_time_col:
            deadline_time_input = st.time_input(
                "締切時刻",
                value=task.deadline.time() if task and task.deadline else time(18, 0)
            )
        
        recurrence_rule = st.selectbox(
            "繰り返し",
            options=[None] + RECURRENCE_RULES,
            format_func=lambda x: "なし" if x is None else {
                "daily": "毎日", "weekly": "毎週", "monthly": "毎月"
            }[x],
            index=RECURRENCE_RULES.index(task.recurrence_rule) + 1 if task and task.recurrence_rule else 0
        )
        
        recurrence_end = None
        if recurrence_rule:
            recurrence_end = st.date_input(
                "繰り返し終了日",
                value=task.recurrence_end.date() if task and task.recurrence_end else None
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("保存" if is_edit else "作成")
        
        with col2:
            if st.form_submit_button("キャンセル"):
                st.session_state.show_task_form = False
                if 'edit_task_id' in st.session_state:
                    del st.session_state.edit_task_id
                st.rerun()
        
        if submitted:
            if not title.strip():
                st.error("タイトルは必須です")
                return
            
            due_datetime = None
            if due_date_input:
                due_datetime = datetime.combine(due_date_input, due_time_input)
            
            deadline_datetime = None
            if deadline_date_input:
                deadline_datetime = datetime.combine(deadline_date_input, deadline_time_input)
            
            recurrence_end_datetime = None
            if recurrence_end:
                recurrence_end_datetime = datetime.combine(recurrence_end, time(23, 59, 59))
            
            try:
                if is_edit:
                    task_update = TaskUpdate(
                        title=title.strip(),
                        description=description.strip() if description and description.strip() else None,
                        priority=priority,
                        status=status,
                        category_id=category_id,
                        due_date=due_datetime,
                        deadline=deadline_datetime,
                        recurrence_rule=recurrence_rule,
                        recurrence_end=recurrence_end_datetime
                    )
                    task_service.update_task(task_id, task_update)
                    st.success("タスクを更新しました")
                else:
                    task_create = TaskCreate(
                        title=title.strip(),
                        description=description.strip() if description and description.strip() else None,
                        priority=priority,
                        status=status,
                        category_id=category_id,
                        due_date=due_datetime,
                        deadline=deadline_datetime,
                        recurrence_rule=recurrence_rule,
                        recurrence_end=recurrence_end_datetime
                    )
                    task_service.create_task(task_create)
                    st.success("タスクを作成しました")
                
                st.session_state.show_task_form = False
                if 'edit_task_id' in st.session_state:
                    del st.session_state.edit_task_id
                st.rerun()
                
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")


def render_quick_add():
    st.subheader("🚀 クイック追加")
    
    with st.form("quick_add_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            quick_title = st.text_input(
                "タスクタイトル",
                placeholder="新しいタスクを入力...",
                label_visibility="collapsed"
            )
        
        with col2:
            quick_submitted = st.form_submit_button("追加")
        
        if quick_submitted and quick_title.strip():
            try:
                task_create = TaskCreate(title=quick_title.strip())
                task_service.create_task(task_create)
                st.success("タスクを追加しました")
                st.rerun()
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")