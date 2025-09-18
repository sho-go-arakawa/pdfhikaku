from typing import List, Optional
from ..models import Category, CategoryCreate, CategoryUpdate
from ..data import category_repo


class CategoryService:
    def create_category(self, category_data: CategoryCreate) -> Category:
        return category_repo.create(category_data)
    
    def get_category_by_id(self, category_id: int) -> Optional[Category]:
        return category_repo.get_by_id(category_id)
    
    def get_all_categories(self) -> List[Category]:
        return category_repo.get_all()
    
    def update_category(self, category_id: int, category_update: CategoryUpdate) -> Optional[Category]:
        return category_repo.update(category_id, category_update)
    
    def delete_category(self, category_id: int) -> bool:
        return category_repo.delete(category_id)


category_service = CategoryService()