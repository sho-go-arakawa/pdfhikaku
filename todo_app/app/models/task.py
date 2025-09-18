from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class TaskBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    priority: int = Field(default=1, ge=0, le=2)
    status: str = Field(default="todo", pattern="^(todo|doing|done)$")
    category_id: Optional[int] = None
    parent_id: Optional[int] = None
    recurrence_rule: Optional[str] = Field(None, pattern="^(daily|weekly|monthly)$")
    recurrence_end: Optional[datetime] = None


class TaskCreate(TaskBase):
    pass


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    priority: Optional[int] = Field(None, ge=0, le=2)
    status: Optional[str] = Field(None, pattern="^(todo|doing|done)$")
    category_id: Optional[int] = None
    parent_id: Optional[int] = None
    recurrence_rule: Optional[str] = Field(None, pattern="^(daily|weekly|monthly)$")
    recurrence_end: Optional[datetime] = None


class Task(TaskBase):
    id: int
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TaskWithDetails(Task):
    category_name: Optional[str] = None
    parent_title: Optional[str] = None
    subtasks: List['Task'] = []
    tags: List[str] = []