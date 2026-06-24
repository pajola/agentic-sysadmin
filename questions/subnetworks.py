from typing import List, Type, Optional
from pydantic import BaseModel, Field
from .base_question import BaseQuestion
import ipaddress

class SubnetworksResponse(BaseModel):
    """Model for network subnets response"""
    subnets: List[str] = Field(
        description="List of subnets in CIDR notation"
    )
    count: int = Field(
        description="Total number of subnets in the network"
    )

class SubnetworksQuestion(BaseQuestion):
    """Question plugin to identify all subnets in the network"""
    
    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)
    
    @property
    def question_text(self) -> str:
        return "List all unique subnets in the network (in CIDR notation) without duplicates or sub-subnets contained within a larger subnet."
    
    @staticmethod
    def output_model() -> Type[BaseModel]:
        return SubnetworksResponse
    
    def get_ground_truth(self) -> BaseModel:
        machines_ips = self._kathara.get_machines_ips()
        
        subnets = set()
        for machine_name, ips in machines_ips.items():
            for ip, cidr, is_special in ips:
                if is_special:
                    continue
                network = ipaddress.ip_network(f"{ip}/{cidr}", strict=False)
                subnets.add(f"{str(network.network_address)}/{cidr}")
        
        subnet_list = list(subnets)
        
        return SubnetworksResponse(
            subnets=subnet_list,
            count=len(subnet_list)
        )