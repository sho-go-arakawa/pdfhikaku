from datetime import datetime
from typing import List, Optional, Dict, Any
from ..models import Task, TaskCreate, TaskUpdate, Category, CategoryCreate, CategoryUpdate
from ..models import Tag, TagCreate, TagUpdate
from .database import db


class TaskRepository:
    def create(self, task: TaskCreate) -> Task:
        # Get next ID
        next_id_query = "SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM tasks"
        next_id_result = db.execute_query_one(next_id_query)
        next_id = next_id_result['next_id']
        
        query = """
        INSERT INTO tasks (id, title, description, due_date, deadline, priority, status, 
                          category_id, parent_id, recurrence_rule, recurrence_end, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        params = (
            next_id, task.title, task.description, task.due_date, task.deadline, 
            task.priority, task.status, task.category_id, task.parent_id,
            task.recurrence_rule, task.recurrence_end
        )
        db.execute_non_query(query, params)
        return self.get_by_id(next_id)
    
    def get_by_id(self, task_id: int) -> Optional[Task]:
        query = "SELECT * FROM tasks WHERE id = ?"
        result = db.execute_query_one(query, (task_id,))
        return Task(**result) if result else None
    
    def get_all(self, status: Optional[str] = None, category_id: Optional[int] = None,
                parent_id: Optional[int] = None, limit: Optional[int] = None) -> List[Task]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if category_id is not None:
            query += " AND category_id = ?"
            params.append(category_id)
        
        if parent_id is not None:
            query += " AND parent_id = ?"
            params.append(parent_id)
        elif parent_id is None:
            query += " AND parent_id IS NULL"
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = db.execute_query(query, tuple(params) if params else None)
        return [Task(**result) for result in results]
    
    def update(self, task_id: int, task_update: TaskUpdate) -> Optional[Task]:
        update_data = task_update.dict(exclude_unset=True)
        if not update_data:
            return self.get_by_id(task_id)
        
        update_data['updated_at'] = datetime.utcnow()
        
        if 'status' in update_data and update_data['status'] == 'done':
            update_data['completed_at'] = datetime.utcnow()
        elif 'status' in update_data and update_data['status'] != 'done':
            update_data['completed_at'] = None
        
        set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
        query = f"UPDATE tasks SET {set_clause} WHERE id = ?"
        params = tuple(update_data.values()) + (task_id,)
        
        db.execute_non_query(query, params)
        return self.get_by_id(task_id)
    
    def delete(self, task_id: int) -> bool:
        query = "DELETE FROM tasks WHERE id = ?"
        affected_rows = db.execute_non_query(query, (task_id,))
        return affected_rows > 0
    
    def get_overdue_tasks(self) -> List[Task]:
        query = """
        SELECT * FROM tasks 
        WHERE (deadline < ? OR due_date < ?) 
        AND status != 'done'
        ORDER BY COALESCE(deadline, due_date)
        """
        now = datetime.utcnow()
        results = db.execute_query(query, (now, now))
        return [Task(**result) for result in results]
    
    def search(self, search_term: str) -> List[Task]:
        query = """
        SELECT * FROM tasks 
        WHERE title ILIKE ? OR description ILIKE ?
        ORDER BY created_at DESC
        """
        term = f"%{search_term}%"
        results = db.execute_query(query, (term, term))
        return [Task(**result) for result in results]


class CategoryRepository:
    def create(self, category: CategoryCreate) -> Category:
        # Get next ID
        next_id_query = "SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM categories"
        next_id_result = db.execute_query_one(next_id_query)
        next_id = next_id_result['next_id']
        
        query = """
        INSERT INTO categories (id, name, description, color, created_at, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        params = (next_id, category.name, category.description, category.color)
        db.execute_non_query(query, params)
        return self.get_by_id(next_id)
    
    def get_by_id(self, category_id: int) -> Optional[Category]:
        query = "SELECT * FROM categories WHERE id = ?"
        result = db.execute_query_one(query, (category_id,))
        return Category(**result) if result else None
    
    def get_all(self) -> List[Category]:
        query = "SELECT * FROM categories ORDER BY name"
        results = db.execute_query(query)
        return [Category(**result) for result in results]
    
    def update(self, category_id: int, category_update: CategoryUpdate) -> Optional[Category]:
        update_data = category_update.dict(exclude_unset=True)
        if not update_data:
            return self.get_by_id(category_id)
        
        update_data['updated_at'] = datetime.utcnow()
        
        set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
        query = f"UPDATE categories SET {set_clause} WHERE id = ?"
        params = tuple(update_data.values()) + (category_id,)
        
        db.execute_non_query(query, params)
        return self.get_by_id(category_id)
    
    def delete(self, category_id: int) -> bool:
        query = "DELETE FROM categories WHERE id = ?"
        affected_rows = db.execute_non_query(query, (category_id,))
        return affected_rows > 0


class TagRepository:
    def create(self, tag: TagCreate) -> Tag:
        # Get next ID
        next_id_query = "SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM tags"
        next_id_result = db.execute_query_one(next_id_query)
        next_id = next_id_result['next_id']
        
        query = """
        INSERT INTO tags (id, name, color, created_at, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        params = (next_id, tag.name, tag.color)
        db.execute_non_query(query, params)
        return self.get_by_id(next_id)
    
    def get_by_id(self, tag_id: int) -> Optional[Tag]:
        query = "SELECT * FROM tags WHERE id = ?"
        result = db.execute_query_one(query, (tag_id,))
        return Tag(**result) if result else None
    
    def get_all(self) -> List[Tag]:
        query = "SELECT * FROM tags ORDER BY name"
        results = db.execute_query(query)
        return [Tag(**result) for result in results]
    
    def update(self, tag_id: int, tag_update: TagUpdate) -> Optional[Tag]:
        update_data = tag_update.dict(exclude_unset=True)
        if not update_data:
            return self.get_by_id(tag_id)
        
        update_data['updated_at'] = datetime.utcnow()
        
        set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
        query = f"UPDATE tags SET {set_clause} WHERE id = ?"
        params = tuple(update_data.values()) + (tag_id,)
        
        db.execute_non_query(query, params)
        return self.get_by_id(tag_id)
    
    def delete(self, tag_id: int) -> bool:
        query = "DELETE FROM tags WHERE id = ?"
        affected_rows = db.execute_non_query(query, (tag_id,))
        return affected_rows > 0
    
    def get_by_task(self, task_id: int) -> List[Tag]:
        query = """
        SELECT t.* FROM tags t
        JOIN task_tags tt ON t.id = tt.tag_id
        WHERE tt.task_id = ?
        ORDER BY t.name
        """
        results = db.execute_query(query, (task_id,))
        return [Tag(**result) for result in results]
    
    def add_to_task(self, task_id: int, tag_id: int) -> bool:
        query = """
        INSERT INTO task_tags (task_id, tag_id)
        VALUES (?, ?)
        ON CONFLICT DO NOTHING
        """
        affected_rows = db.execute_non_query(query, (task_id, tag_id))
        return affected_rows > 0
    
    def remove_from_task(self, task_id: int, tag_id: int) -> bool:
        query = "DELETE FROM task_tags WHERE task_id = ? AND tag_id = ?"
        affected_rows = db.execute_non_query(query, (task_id, tag_id))
        return affected_rows > 0


task_repo = TaskRepository()
category_repo = CategoryRepository()
tag_repo = TagRepository()