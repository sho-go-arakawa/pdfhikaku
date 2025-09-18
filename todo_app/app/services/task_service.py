from datetime import datetime, timedelta
from typing import List, Optional
from ..models import Task, TaskCreate, TaskUpdate, TaskWithDetails
from ..data import task_repo, category_repo, tag_repo
from ..settings import PRIORITY_LEVELS


class TaskService:
    def create_task(self, task_data: TaskCreate) -> Task:
        task = task_repo.create(task_data)
        
        if task.recurrence_rule:
            self._create_recurring_task_if_needed(task)
        
        return task
    
    def get_task_by_id(self, task_id: int) -> Optional[TaskWithDetails]:
        task = task_repo.get_by_id(task_id)
        if not task:
            return None
        
        return self._enrich_task_with_details(task)
    
    def get_all_tasks(self, status: Optional[str] = None, 
                     category_id: Optional[int] = None,
                     parent_id: Optional[int] = None,
                     limit: Optional[int] = None) -> List[TaskWithDetails]:
        tasks = task_repo.get_all(status, category_id, parent_id, limit)
        return [self._enrich_task_with_details(task) for task in tasks]
    
    def update_task(self, task_id: int, task_update: TaskUpdate) -> Optional[TaskWithDetails]:
        task = task_repo.update(task_id, task_update)
        if not task:
            return None
        
        if task.recurrence_rule and task.status == 'done':
            self._create_recurring_task_if_needed(task)
        
        return self._enrich_task_with_details(task)
    
    def delete_task(self, task_id: int) -> bool:
        return task_repo.delete(task_id)
    
    def toggle_task_status(self, task_id: int) -> Optional[TaskWithDetails]:
        task = task_repo.get_by_id(task_id)
        if not task:
            return None
        
        new_status = 'done' if task.status != 'done' else 'todo'
        update_data = TaskUpdate(status=new_status)
        
        return self.update_task(task_id, update_data)
    
    def get_overdue_tasks(self) -> List[TaskWithDetails]:
        tasks = task_repo.get_overdue_tasks()
        return [self._enrich_task_with_details(task) for task in tasks]
    
    def search_tasks(self, search_term: str) -> List[TaskWithDetails]:
        tasks = task_repo.search(search_term)
        return [self._enrich_task_with_details(task) for task in tasks]
    
    def get_tasks_by_priority(self, priority: int) -> List[TaskWithDetails]:
        tasks = task_repo.get_all()
        filtered_tasks = [task for task in tasks if task.priority == priority]
        return [self._enrich_task_with_details(task) for task in filtered_tasks]
    
    def get_tasks_due_soon(self, days_ahead: int = 7) -> List[TaskWithDetails]:
        tasks = task_repo.get_all()
        threshold = datetime.utcnow() + timedelta(days=days_ahead)
        
        due_soon = []
        for task in tasks:
            if task.status != 'done':
                if task.due_date and task.due_date <= threshold:
                    due_soon.append(task)
                elif task.deadline and task.deadline <= threshold:
                    due_soon.append(task)
        
        return [self._enrich_task_with_details(task) for task in due_soon]
    
    def add_tag_to_task(self, task_id: int, tag_id: int) -> bool:
        return tag_repo.add_to_task(task_id, tag_id)
    
    def remove_tag_from_task(self, task_id: int, tag_id: int) -> bool:
        return tag_repo.remove_from_task(task_id, tag_id)
    
    def _enrich_task_with_details(self, task: Task) -> TaskWithDetails:
        category_name = None
        if task.category_id:
            category = category_repo.get_by_id(task.category_id)
            if category:
                category_name = category.name
        
        parent_title = None
        if task.parent_id:
            parent_task = task_repo.get_by_id(task.parent_id)
            if parent_task:
                parent_title = parent_task.title
        
        subtasks = task_repo.get_all(parent_id=task.id)
        
        tags = tag_repo.get_by_task(task.id)
        tag_names = [tag.name for tag in tags]
        
        return TaskWithDetails(
            **task.dict(),
            category_name=category_name,
            parent_title=parent_title,
            subtasks=subtasks,
            tags=tag_names
        )
    
    def _create_recurring_task_if_needed(self, completed_task: Task):
        if not completed_task.recurrence_rule or completed_task.status != 'done':
            return
        
        if completed_task.recurrence_end and datetime.utcnow() > completed_task.recurrence_end:
            return
        
        next_due_date = self._calculate_next_due_date(
            completed_task.due_date or completed_task.created_at,
            completed_task.recurrence_rule
        )
        
        next_deadline = None
        if completed_task.deadline:
            next_deadline = self._calculate_next_due_date(
                completed_task.deadline,
                completed_task.recurrence_rule
            )
        
        new_task_data = TaskCreate(
            title=completed_task.title,
            description=completed_task.description,
            due_date=next_due_date,
            deadline=next_deadline,
            priority=completed_task.priority,
            category_id=completed_task.category_id,
            parent_id=completed_task.parent_id,
            recurrence_rule=completed_task.recurrence_rule,
            recurrence_end=completed_task.recurrence_end
        )
        
        task_repo.create(new_task_data)
    
    def _calculate_next_due_date(self, current_date: datetime, recurrence_rule: str) -> datetime:
        if recurrence_rule == 'daily':
            return current_date + timedelta(days=1)
        elif recurrence_rule == 'weekly':
            return current_date + timedelta(weeks=1)
        elif recurrence_rule == 'monthly':
            if current_date.month == 12:
                return current_date.replace(year=current_date.year + 1, month=1)
            else:
                return current_date.replace(month=current_date.month + 1)
        
        return current_date


task_service = TaskService()