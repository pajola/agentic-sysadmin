from typing import Type, Optional, List
from pydantic import BaseModel, Field
from .base_question import BaseQuestion

class CanPingWithoutHopResponse(BaseModel):
    """
    Model for checking if two machines can ping each other directly.
    """
    success: bool = Field(
        description="True if direct communication is possible, False otherwise"
    )

class CanPingWithoutHopQuestion(BaseQuestion):
    """
    Question plugin to determine if two machines can ping each other
    without intermediate hops.
    """
    def __init__(self, m1: str, m2: str, lab_whitelist: Optional[List[str]] = None):
        """
        Initialize with the two machine names.
        
        Args:
            m1 (str): Name of the first machine.
            m2 (str): Name of the second machine.
            lab_whitelist: List of lab names this question should run on.
        """
        super().__init__(lab_whitelist=lab_whitelist)
        self.m1 = m1
        self.m2 = m2

    def cache_key(self) -> str:
        return f"{self.__class__.__name__}::{self.m1}::{self.m2}"

    @property
    def question_text(self) -> str:
        return f"Can machine '{self.m1}' ping machine '{self.m2}' directly without intermediate hops?"
    
    @staticmethod
    def output_model() -> Type[BaseModel]:
        return CanPingWithoutHopResponse
    
    def get_ground_truth(self) -> BaseModel:
        """
        Determine if m1 can ping m2 without intermediate hops.
        """
        result = self._kathara.can_ping_without_hop(self.m1, self.m2)
        return CanPingWithoutHopResponse(success=result)