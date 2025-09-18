from typing import List, Optional
from ..models import Tag, TagCreate, TagUpdate
from ..data import tag_repo


class TagService:
    def create_tag(self, tag_data: TagCreate) -> Tag:
        return tag_repo.create(tag_data)
    
    def get_tag_by_id(self, tag_id: int) -> Optional[Tag]:
        return tag_repo.get_by_id(tag_id)
    
    def get_all_tags(self) -> List[Tag]:
        return tag_repo.get_all()
    
    def update_tag(self, tag_id: int, tag_update: TagUpdate) -> Optional[Tag]:
        return tag_repo.update(tag_id, tag_update)
    
    def delete_tag(self, tag_id: int) -> bool:
        return tag_repo.delete(tag_id)
    
    def get_tags_by_task(self, task_id: int) -> List[Tag]:
        return tag_repo.get_by_task(task_id)


tag_service = TagService()