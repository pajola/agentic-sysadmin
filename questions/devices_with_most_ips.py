from typing import List, Type, Optional
from pydantic import BaseModel, Field
from .base_question import BaseQuestion

class DevicesWithMostIPsResponse(BaseModel):
    """
    Model for devices with the most IP addresses.
    """
    devices: List[str] = Field(
        description="List of devices that have the most IP addresses configured. "
                    "If there is a tie, all devices with the maximum count are included."
    )
    max_ips: int = Field(
        description="Maximum number of IP addresses configured on a single device."
    )

class DevicesWithMostIPsQuestion(BaseQuestion):
    """
    Question plugin to identify the devices having the highest number of IP addresses.
    """
    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)

    @property
    def question_text(self) -> str:
        return "Which devices in the network have the most IP addresses configured?"

    @staticmethod
    def output_model() -> Type[BaseModel]:
        return DevicesWithMostIPsResponse

    def get_ground_truth(self) -> BaseModel:
        # Use the higher-level method of the KatharaClient.
        machines_ips = self._kathara.get_machines_ips()
        filtered_ips = {m: [ (ip, cidr) for ip, cidr, is_special in ips if not is_special ] for m, ips in machines_ips.items()}

        # Create a mapping from device name to the count of IP addresses it has.
        device_count = {machine: len(ips) for machine, ips in filtered_ips.items()}
        if device_count:
            max_ips = max(device_count.values())
            most_devices = [device for device, count in device_count.items() if count == max_ips]
        else:
            max_ips = 0
            most_devices = []
        return DevicesWithMostIPsResponse(devices=most_devices, max_ips=max_ips)