from typing import Optional, Type, List
from pydantic import BaseModel, Field
from .base_question import BaseQuestion
import ipaddress

class CommonSubnetworkResponse(BaseModel):
    """Model for common subnet between two devices response"""
    device1: str = Field(description="Name of the first device")
    device2: str = Field(description="Name of the second device")
    common_subnet: Optional[str] = Field(
        default=None,
        description="Common subnet in CIDR notation if exists, null otherwise"
    )

class CommonSubnetworkQuestion(BaseQuestion):
    """Question plugin to find common subnet between two devices"""
    
    def __init__(self, m1: str, m2: str, lab_whitelist: Optional[List[str]] = None):
        """
        Initialize with device names and optional lab whitelist.
        
        Args:
            m1 (str): Name of the first device
            m2 (str): Name of the second device
            lab_whitelist: List of lab names this question should run on
        """
        super().__init__(lab_whitelist=lab_whitelist)
        self._device1 = m1
        self._device2 = m2

    def cache_key(self) -> str:
        return f"{self.__class__.__name__}::{self._device1}::{self._device2}"

    @property
    def question_text(self) -> str:
        return f"Are devices {self._device1} and {self._device2} directly connected? If so, on which subnet?"
    
    @staticmethod
    def output_model() -> Type[BaseModel]:
        return CommonSubnetworkResponse
    
    def get_ground_truth(self) -> BaseModel:
        machines_ips = self._kathara.get_machines_ips()
        
        # Validate device names exist
        if self._device1 not in machines_ips or self._device2 not in machines_ips:
            missing = []
            if self._device1 not in machines_ips:
                missing.append(self._device1)
            if self._device2 not in machines_ips:
                missing.append(self._device2)
            raise ValueError(f"Device(s) not found: {', '.join(missing)}")
        
        # Find common subnet
        common_subnet = None
        for ip1, cidr1, is_special1 in machines_ips[self._device1]:
            if is_special1:
                continue
            net1 = ipaddress.ip_network(f"{ip1}/{cidr1}", strict=False)
            for ip2, cidr2, is_special2 in machines_ips[self._device2]:
                if is_special2:
                    continue
                net2 = ipaddress.ip_network(f"{ip2}/{cidr2}", strict=False)
                if net1 == net2:
                    common_subnet = str(net1)
                    break
            if common_subnet:
                break
        
        return CommonSubnetworkResponse(
            device1=self._device1,
            device2=self._device2,
            common_subnet=common_subnet
        )