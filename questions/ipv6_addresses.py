from typing import List, Tuple, Type, Optional
from pydantic import BaseModel, Field
from .base_question import BaseQuestion
import ipaddress

class IPv6AddressItem(BaseModel):
    """Model for an individual IPv6 address entry"""
    address: str = Field(description="IPv6 address with CIDR notation")
    device: str = Field(description="Name of the device with this IPv6 address")

class IPv6AddressesResponse(BaseModel):
    """Model for IPv6 addresses response"""
    addresses: List[IPv6AddressItem] = Field(
        description="List of IPv6 addresses in the network"
    )
    count: int = Field(
        description="Total number of IPv6 addresses found"
    )

class IPv6AddressesQuestion(BaseQuestion):
    """Question plugin to list all IPv6 addresses in the network"""
    
    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)
    
    @property
    def question_text(self) -> str:
        return "Which IPv6 addresses are configured in the network, and on which devices?"
    
    @staticmethod
    def output_model() -> Type[BaseModel]:
        return IPv6AddressesResponse
    
    def get_ground_truth(self) -> BaseModel:
        machines_ips = self._kathara.get_machines_ips()
        
        ipv6_addresses = []
        for machine_name, ips in machines_ips.items():
            for ip, cidr, _ in ips:
                if ipaddress.ip_address(ip).version == 6:
                    ipv6_addresses.append(
                        IPv6AddressItem(
                            address=f"{ip}/{cidr}",
                            device=machine_name
                        )
                    )
        
        return IPv6AddressesResponse(
            addresses=ipv6_addresses,
            count=len(ipv6_addresses)
        )