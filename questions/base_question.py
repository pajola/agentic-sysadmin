from abc import ABC, abstractmethod
from typing import Type, TYPE_CHECKING, Optional, List
from pydantic import BaseModel
from deepdiff import DeepDiff

if TYPE_CHECKING:
    from core.kathara_client import KatharaClient

class BaseQuestion(ABC):
    """
    Abstract base class for all question plugins.
    """

    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        """
        Initialize the question with an optional lab whitelist.
        
        Args:
            lab_whitelist: List of lab names this question should run on.
                          If None, the question runs on all labs.
        """
        self._kathara = None
        self.lab_whitelist = lab_whitelist

    def applies_to_lab(self, lab_name: str) -> bool:
        """
        Check if this question should be applied to the given lab.
        
        Args:
            lab_name: The name of the lab to check
            
        Returns:
            True if the question should run on this lab, False otherwise
        """
        if self.lab_whitelist is None:
            return True  # Run on all labs if no whitelist specified
        return lab_name in self.lab_whitelist

    @property
    @abstractmethod
    def question_text(self) -> str:
        """
        Returns the textual question to be posed to the LLM.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self) -> BaseModel:
        """
        Compute and return the ground truth, as an instance of a Pydantic model,
        extracted directly from Kathara via dedicated functions.
        """
        pass

    @staticmethod
    @abstractmethod
    def output_model() -> Type[BaseModel]:
        """
        Return the Pydantic model that defines the expected output structure,
        which can be passed to Instructor as the response_model.
        """
        pass

    def cache_key(self) -> str:
        """
        Stable identifier for caching this question's ground truth in
        labs/<lab>/.ground_truth_cache.json.

        Default: the class name. Parametric questions (e.g. CanPing with
        m1/m2 endpoints) must override this to include the parameters,
        otherwise different parameter combinations would clobber each other.
        """
        return self.__class__.__name__

    def verify(self, ground_truth: BaseModel, response: BaseModel) -> dict:
        """
        Compare a solver's `response` against the `ground_truth`.

        Returns a diff dict: an empty dict means the answer is correct, while a
        non-empty dict describes the mismatch (stored as `verification_diff`).

        The default is a structural comparison via DeepDiff. Questions whose
        answer is not unique (e.g. traceroute, where several paths are equally
        valid) should override this with domain-specific validation.

        Args:
            ground_truth (BaseModel): The expected answer.
            response (BaseModel): The solver's answer.

        Returns:
            dict: Empty if correct, otherwise the diff describing the mismatch.
        """
        diff = DeepDiff(
            ground_truth.model_dump(),
            response.model_dump(),
            ignore_order=True,
        )
        return diff.to_dict()

    def inject_client(self, client: "KatharaClient") -> None:
        """
        Injects a KatharaClient at runtime.

        This method allows the AnalysisEngine to provide a shared KatharaClient
        to each question.
        
        Args:
            client (KatharaClient): The Kathara client instance to use

        """
        self._kathara = client