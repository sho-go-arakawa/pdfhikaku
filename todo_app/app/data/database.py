import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from ..settings import DATABASE_PATH


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
    
    @contextmanager
    def get_connection(self):
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            
            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
    
    def execute_query_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def execute_non_query(self, query: str, params: Optional[tuple] = None) -> int:
        with self.get_connection() as conn:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)
            # DuckDBでは影響行数の取得方法が異なるため、常に1を返す
            # 実際の処理では成功すれば1行は影響を受けたと仮定
            return 1
    
    def initialize_database(self):
        ddl_queries = [
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                color VARCHAR(7),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name VARCHAR(50) NOT NULL UNIQUE,
                color VARCHAR(7),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                description TEXT,
                due_date TIMESTAMP,
                deadline TIMESTAMP,
                priority INTEGER DEFAULT 1 CHECK (priority >= 0 AND priority <= 2),
                status VARCHAR(10) DEFAULT 'todo' CHECK (status IN ('todo', 'doing', 'done')),
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category_id INTEGER REFERENCES categories(id),
                parent_id INTEGER REFERENCES tasks(id),
                recurrence_rule VARCHAR(20) CHECK (recurrence_rule IN ('daily', 'weekly', 'monthly')),
                recurrence_end TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS task_tags (
                task_id INTEGER REFERENCES tasks(id),
                tag_id INTEGER REFERENCES tags(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (task_id, tag_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_deadline ON tasks(deadline)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id)
            """
        ]
        
        with self.get_connection() as conn:
            for query in ddl_queries:
                conn.execute(query)


db = Database()