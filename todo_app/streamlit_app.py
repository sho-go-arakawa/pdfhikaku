import streamlit as st
import atexit
from datetime import datetime
from app.data import db
from app.settings import APP_TITLE, APP_VERSION
from app.scheduler import reminder_service, get_upcoming_reminders
from app.ui import (
    render_task_list, render_task_form, render_quick_add,
    render_category_management, render_tag_management
)
from app.charts import render_burndown_chart


def initialize_app():
    """アプリケーションの初期化"""
    # データベースの初期化
    db.initialize_database()
    
    # リマインダーサービスの開始
    reminder_service.start()
    
    # アプリ終了時にリマインダーサービスを停止
    def cleanup():
        reminder_service.stop()
    
    atexit.register(cleanup)


def render_sidebar():
    """サイドバーの描画"""
    with st.sidebar:
        st.title(f"{APP_TITLE} {APP_VERSION}")
        
        # ナビゲーション
        pages = {
            "📋 タスク一覧": "task_list",
            "➕ タスク作成": "task_create", 
            "📁 カテゴリ管理": "category_management",
            "🏷️ タグ管理": "tag_management",
            "📈 チャート": "burndown_chart"
        }
        
        selected_page = st.radio("ナビゲーション", list(pages.keys()))
        st.session_state.current_page = pages[selected_page]
        
        st.divider()
        
        # クイック追加
        render_quick_add()
        
        st.divider()
        
        # 今後のリマインダー
        render_upcoming_reminders()
        
        st.divider()
        
        # 通知センター
        render_notifications()


def render_upcoming_reminders():
    """今後のリマインダー表示"""
    st.subheader("⏰ 今後のリマインダー")
    
    upcoming = get_upcoming_reminders(days_ahead=3)
    
    if not upcoming:
        st.info("今後3日間のリマインダーはありません")
        return
    
    for reminder in upcoming[:5]:  # 最大5件表示
        task = reminder['task']
        reminder_time = reminder['reminder_time']
        target_time = reminder['target_time']
        reminder_type = reminder['type']
        hours_before = reminder['hours_before']
        
        with st.container():
            st.markdown(f"**{task.title}**")
            st.markdown(f"📅 {reminder_type}: {target_time.strftime('%m/%d %H:%M')}")
            st.markdown(f"🔔 通知: {reminder_time.strftime('%m/%d %H:%M')} ({hours_before}h前)")
            st.markdown("---")


def render_notifications():
    """通知センターの描画"""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    notifications = st.session_state.notifications
    unread_count = len([n for n in notifications if not n['read']])
    
    if unread_count > 0:
        st.subheader(f"🔔 通知 ({unread_count})")
        
        for notification in notifications[-3:]:  # 最新3件を表示
            if not notification['read']:
                with st.container():
                    st.markdown(f"**{notification['task_title']}**")
                    time_text = f"{notification['hours_before']}時間前" if notification['hours_before'] >= 1 else "まもなく"
                    st.markdown(f"⚠️ {notification['deadline_type']}の{time_text}")
                    st.markdown(f"🕒 {notification['target_datetime'].strftime('%m/%d %H:%M')}")
                    
                    if st.button("既読", key=f"read_{notification['id']}"):
                        notification['read'] = True
                        st.rerun()
                    
                    st.markdown("---")
    else:
        st.subheader("🔔 通知")
        st.info("新しい通知はありません")


def render_main_content():
    """メインコンテンツの描画"""
    current_page = st.session_state.get('current_page', 'task_list')
    
    # タスクフォームの表示状態
    if st.session_state.get('show_task_form', False):
        render_task_form()
        return
    
    # ページ別のコンテンツ描画
    if current_page == 'task_list':
        render_task_list()
        
        # 新規作成ボタン
        if st.button("➕ 新しいタスクを作成", type="primary"):
            st.session_state.show_task_form = True
            st.session_state.edit_task_id = None
            st.rerun()
    
    elif current_page == 'task_create':
        render_task_form()
    
    elif current_page == 'category_management':
        render_category_management()
    
    elif current_page == 'tag_management':
        render_tag_management()
    
    elif current_page == 'burndown_chart':
        render_burndown_chart()


def render_stats_header():
    """統計情報ヘッダーの描画"""
    from app.services import task_service
    
    # 統計データを取得
    all_tasks = task_service.get_all_tasks()
    todo_tasks = [t for t in all_tasks if t.status == 'todo']
    doing_tasks = [t for t in all_tasks if t.status == 'doing']
    done_tasks = [t for t in all_tasks if t.status == 'done']
    overdue_tasks = task_service.get_overdue_tasks()
    
    # ヘッダー統計表示
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("全タスク", len(all_tasks))
    
    with col2:
        st.metric("未着手", len(todo_tasks))
    
    with col3:
        st.metric("進行中", len(doing_tasks))
    
    with col4:
        st.metric("完了", len(done_tasks))
    
    with col5:
        st.metric("期限超過", len(overdue_tasks), delta=None if len(overdue_tasks) == 0 else -len(overdue_tasks))


def main():
    """メイン関数"""
    # Streamlitページ設定
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # アプリケーション初期化
    initialize_app()
    
    # サイドバー描画
    render_sidebar()
    
    # メインコンテンツ
    with st.container():
        # 統計情報ヘッダー
        render_stats_header()
        
        st.divider()
        
        # メインコンテンツ描画
        render_main_content()


if __name__ == "__main__":
    main()