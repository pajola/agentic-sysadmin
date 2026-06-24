from typing import List, Type, Optional
from pydantic import BaseModel, Field
from .base_question import BaseQuestion

class MultipleIPsResponse(BaseModel):
    """Model for devices with multiple IP addresses response"""
    devices: List[str] = Field(
        description="List of device names that have multiple IP addresses configured"
    )
    count: int = Field(
        description="Total number of devices with multiple IP addresses"
    )

class DevicesWithMultipleIPsQuestion(BaseQuestion):
    """Question plugin to identify devices with multiple IP addresses"""
    
    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)
    
    @property
    def question_text(self) -> str:
        return "Which devices in the network have multiple IP addresses configured?"
    
    @staticmethod
    def output_model() -> Type[BaseModel]:
        return MultipleIPsResponse
    
    def get_ground_truth(self) -> BaseModel:
        machines_ips = self._kathara.get_machines_ips()
        filtered_ips = {m: [ (ip, cidr) for ip, cidr, is_special in ips if not is_special ] for m, ips in machines_ips.items()}

        devices = []
        for machine_name, ips in filtered_ips.items():
            if len(ips) > 1:
                devices.append(machine_name)
        
        return MultipleIPsResponse(
            devices=devices,
            count=len(devices)
        )