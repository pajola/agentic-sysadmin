from typing import List, Type, Optional
from pydantic import BaseModel, Field, model_validator
from .base_question import BaseQuestion

class TracerouteResponse(BaseModel):
    """
    Model for the traceroute result between two machines.
    
    The trace should represent the complete path from source to destination,
    including both the source machine (first) and destination machine (last).
    If the destination is unreachable, trace will be None.
    """
    trace: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of all machine names traversed in the traceroute path. "
                    "The first element must be the source machine, and the last element "
                    "must be the destination machine. Include all intermediate hops. "
                    "Will be None if destination is unreachable.",
        examples=[
            ["source_machine", "intermediate_router", "destination_machine"],
            ["client1", "router1", "router2", "router3", "client2"]
        ]
    )

class TracerouteQuestion(BaseQuestion):
    """
    Question plugin to obtain the traceroute between two machines.
    """
    def __init__(self, m1: str, m2: str, lab_whitelist: Optional[List[str]] = None):
        """
        Initialize with the two machine names.
        
        Args:
            m1 (str): Name of the source machine.
            m2 (str): Name of the destination machine.
            lab_whitelist: List of lab names this question should run on.
        """
        super().__init__(lab_whitelist=lab_whitelist)
        self.m1 = m1
        self.m2 = m2

    def cache_key(self) -> str:
        return f"{self.__class__.__name__}::{self.m1}::{self.m2}"

    @property
    def question_text(self) -> str:
        return f"What is the traceroute from '{self.m1}' to '{self.m2}'?"

    @staticmethod
    def output_model() -> Type[BaseModel]:
        return TracerouteResponse

    def get_ground_truth(self) -> BaseModel:
        """        
        Returns:
            TracerouteResponse: Response model containing the list of hops or unreachable status.
        """
        trace = self._kathara.traceroute_names(self.m1, self.m2)
        
        if trace is None:
            # Destination is unreachable
            return TracerouteResponse()
        else:
            # Add the name of the first device at the start of the trace list
            trace.insert(0, self.m1)
            return TracerouteResponse(trace=trace)

    def verify(self, ground_truth: BaseModel, response: BaseModel) -> dict:
        """
        Grade a traceroute answer by forwarding-plane validity rather than by
        exact match against a single sampled path.

        There is no unique correct traceroute: with ECMP or multiple
        destination interfaces several paths are equally valid. We accept any
        path the network would actually forward from m1 to m2 (validated
        against the live FIBs); the cached ground-truth trace is kept only for
        reference in the results.

        Returns an empty dict when the answer is correct, otherwise a diff
        describing the mismatch.
        """
        if response.trace is None:
            # A None answer is correct only if the destination is unreachable.
            if ground_truth.trace is None:
                return {}
            return {"traceroute": "expected a reachable path, got unreachable"}

        if self._kathara.is_valid_traceroute(self.m1, self.m2, response.trace):
            return {}
        return {"traceroute": {"invalid_forwarding_path": response.trace}}