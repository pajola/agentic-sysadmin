from typing import Type, Optional, List
from pydantic import BaseModel, Field
from .base_question import BaseQuestion

class CountNodesResponse(BaseModel):
    """Model for the count of nodes in the network."""
    count: int = Field(
        description="Total number of nodes in the network"
    )

class CountNodesQuestion(BaseQuestion):
    """Question plugin to count the nodes in the network."""
    
    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)
    
    @property
    def question_text(self) -> str:
        return "What is the total number of nodes in the network?"
    
    @staticmethod
    def output_model() -> Type[BaseModel]:
        return CountNodesResponse
    
    def get_ground_truth(self) -> BaseModel:
        node_count = self._kathara.count_nodes()
        return CountNodesResponse(count=node_count)