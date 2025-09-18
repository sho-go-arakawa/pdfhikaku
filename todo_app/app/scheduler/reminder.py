import logging
from datetime import datetime, timedelta
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from ..services import task_service
from ..settings import NOTIFICATION_HOURS_BEFORE


logger = logging.getLogger(__name__)


class ReminderService:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.is_running = False
    
    def start(self):
        if not self.is_running:
            self.scheduler.add_job(
                func=self.check_reminders,
                trigger=IntervalTrigger(minutes=30),  # 30分ごとに実行
                id='reminder_checker',
                name='Check task reminders',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Reminder scheduler started")
    
    def stop(self):
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Reminder scheduler stopped")
    
    def check_reminders(self):
        """期日・締切のリマインダーをチェックし、通知を送信"""
        try:
            now = datetime.utcnow()
            
            # 完了していないタスクを取得
            tasks = task_service.get_all_tasks(status="todo") + task_service.get_all_tasks(status="doing")
            
            for task in tasks:
                self._check_task_reminders(task, now)
                
        except Exception as e:
            logger.error(f"Error checking reminders: {e}")
    
    def _check_task_reminders(self, task, now: datetime):
        """個別タスクのリマインダーをチェック"""
        # 期日のチェック
        if task.due_date:
            self._check_deadline_reminder(task, task.due_date, now, "期日")
        
        # 締切のチェック
        if task.deadline:
            self._check_deadline_reminder(task, task.deadline, now, "締切")
    
    def _check_deadline_reminder(self, task, target_datetime: datetime, now: datetime, deadline_type: str):
        """指定された日時に対するリマインダーをチェック"""
        for hours_before in NOTIFICATION_HOURS_BEFORE:
            reminder_time = target_datetime - timedelta(hours=hours_before)
            
            # リマインダー時刻から30分以内（スケジューラーの実行間隔を考慮）
            if abs((now - reminder_time).total_seconds()) <= 30 * 60:
                self._send_notification(task, target_datetime, hours_before, deadline_type)
    
    def _send_notification(self, task, target_datetime: datetime, hours_before: int, deadline_type: str):
        """通知を送信（現在はログ出力のみ）"""
        time_text = f"{hours_before}時間前" if hours_before >= 1 else "1時間以内"
        
        message = f"""
🔔 タスクリマインダー
📝 タスク: {task.title}
⏰ {deadline_type}: {target_datetime.strftime('%Y年%m月%d日 %H:%M')}
⚠️  {time_text}の通知です
        """.strip()
        
        # 現在はコンソールに出力（実際のアプリでは、メール、LINE、Slackなどに送信）
        print(message)
        logger.info(f"Reminder sent for task {task.id}: {task.title}")
        
        # Streamlitのセッションステートに通知を追加（UI表示用）
        self._add_notification_to_session(task, target_datetime, hours_before, deadline_type)
    
    def _add_notification_to_session(self, task, target_datetime: datetime, hours_before: int, deadline_type: str):
        """Streamlitセッションに通知を追加"""
        try:
            import streamlit as st
            
            if 'notifications' not in st.session_state:
                st.session_state.notifications = []
            
            notification = {
                'id': f"{task.id}_{deadline_type}_{hours_before}",
                'task_id': task.id,
                'task_title': task.title,
                'target_datetime': target_datetime,
                'hours_before': hours_before,
                'deadline_type': deadline_type,
                'created_at': datetime.utcnow(),
                'read': False
            }
            
            # 重複チェック
            existing_ids = [n['id'] for n in st.session_state.notifications]
            if notification['id'] not in existing_ids:
                st.session_state.notifications.append(notification)
                
                # 古い通知を削除（最新50件まで保持）
                st.session_state.notifications = st.session_state.notifications[-50:]
                
        except Exception as e:
            logger.warning(f"Could not add notification to session: {e}")


def get_upcoming_reminders(days_ahead: int = 7) -> List[dict]:
    """今後数日以内にリマインダーがあるタスクを取得"""
    now = datetime.utcnow()
    end_time = now + timedelta(days=days_ahead)
    
    tasks = task_service.get_all_tasks(status="todo") + task_service.get_all_tasks(status="doing")
    
    upcoming = []
    for task in tasks:
        # 期日のリマインダー
        if task.due_date and now <= task.due_date <= end_time:
            for hours_before in NOTIFICATION_HOURS_BEFORE:
                reminder_time = task.due_date - timedelta(hours=hours_before)
                if reminder_time >= now:
                    upcoming.append({
                        'task': task,
                        'reminder_time': reminder_time,
                        'target_time': task.due_date,
                        'type': '期日',
                        'hours_before': hours_before
                    })
        
        # 締切のリマインダー
        if task.deadline and now <= task.deadline <= end_time:
            for hours_before in NOTIFICATION_HOURS_BEFORE:
                reminder_time = task.deadline - timedelta(hours=hours_before)
                if reminder_time >= now:
                    upcoming.append({
                        'task': task,
                        'reminder_time': reminder_time,
                        'target_time': task.deadline,
                        'type': '締切',
                        'hours_before': hours_before
                    })
    
    # リマインダー時刻でソート
    upcoming.sort(key=lambda x: x['reminder_time'])
    return upcoming


# グローバルインスタンス
reminder_service = ReminderService()