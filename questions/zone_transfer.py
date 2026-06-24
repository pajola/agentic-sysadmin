from typing import List, Type, Optional
from pydantic import BaseModel, Field
from .base_question import BaseQuestion

class ZoneTransferItem(BaseModel):
    device: str = Field(description="DNS server device name")
    zone: str = Field(description="Zone name that allows transfers")
    allowed_hosts: List[str] = Field(description="List of hosts/networks allowed to transfer")

class ZoneTransferResponse(BaseModel):
    vulnerable_servers: List[ZoneTransferItem] = Field(
        description="List of DNS servers that allow zone transfers"
    )
    count: int = Field(description="Total number of DNS servers allowing zone transfers")

class ZoneTransferQuestion(BaseQuestion):
    """Question plugin to identify DNS servers that allow zone transfers."""

    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)

    @property
    def question_text(self) -> str:
        return "Is zone transfer allowed on any DNS server? Which zones and to which hosts?"

    @staticmethod
    def output_model() -> Type[BaseModel]:
        return ZoneTransferResponse

    def get_ground_truth(self) -> BaseModel:
        machines_ips = self._kathara.get_machines_ips()
        vulnerable_servers = []
        
        for machine_name in machines_ips.keys():
            # Check if this machine runs a DNS server
            dns_running = self._check_dns_service(machine_name)
            if not dns_running:
                continue
            
            # Check for zone transfer configuration
            zone_transfers = self._check_zone_transfers(machine_name)
            if zone_transfers:
                vulnerable_servers.extend(zone_transfers)
        
        return ZoneTransferResponse(
            vulnerable_servers=vulnerable_servers,
            count=len(vulnerable_servers)
        )
    
    def _check_dns_service(self, machine_name: str) -> bool:
        """Check if the machine is running a DNS service."""
        # Check for running named/bind9 process
        output = self._kathara.execute_command(machine_name, "pgrep named")
        if output and output.strip():
            return True
        
        # Check for bind9 process
        output = self._kathara.execute_command(machine_name, "pgrep bind9")
        if output and output.strip():
            return True
        
        # Check if systemctl shows named as active
        output = self._kathara.execute_command(machine_name, "systemctl is-active named")
        if output and "active" in output.lower():
            return True
            
        return False
    
    def _check_zone_transfers(self, machine_name: str) -> List[ZoneTransferItem]:
        """Check zone transfer configuration for a DNS server."""
        zone_transfers = []
        
        # Check named.conf files for allow-transfer directives
        config_files = [
            "/etc/bind/named.conf",
            "/etc/named.conf",
            "/etc/bind/named.conf.local",
            "/etc/bind/named.conf.default-zones"
        ]
        
        for config_file in config_files:
            content = self._kathara.execute_command(machine_name, f"cat {config_file}")
            if not content:
                continue
                
            # Parse configuration for allow-transfer directives
            lines = content.splitlines()
            current_zone = None
            in_zone_block = False
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                
                # Track zone blocks
                if line.startswith('zone '):
                    # Extract zone name
                    parts = line.split('"')
                    if len(parts) >= 2:
                        current_zone = parts[1]
                    in_zone_block = True
                    brace_count = line.count('{') - line.count('}')
                elif in_zone_block:
                    brace_count += line.count('{') - line.count('}')
                    
                    # Check for allow-transfer in this zone
                    if 'allow-transfer' in line:
                        # Parse allowed hosts
                        allowed_hosts = self._parse_allow_transfer(line)
                        zone_transfers.append(ZoneTransferItem(
                            device=machine_name,
                            zone=current_zone or "unknown",
                            allowed_hosts=allowed_hosts
                        ))
                    
                    # End of zone block
                    if brace_count <= 0:
                        in_zone_block = False
                        current_zone = None
        
        # Also check for global allow-transfer (outside zone blocks)
        for config_file in config_files:
            content = self._kathara.execute_command(machine_name, f"cat {config_file}")
            if not content:
                continue
                
            lines = content.splitlines()
            in_zone = False
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                
                # Track if we're inside a zone block
                if line.startswith('zone '):
                    in_zone = True
                    brace_count = line.count('{') - line.count('}')
                elif in_zone:
                    brace_count += line.count('{') - line.count('}')
                    if brace_count <= 0:
                        in_zone = False
                elif 'allow-transfer' in line and not in_zone:
                    # Global allow-transfer found
                    allowed_hosts = self._parse_allow_transfer(line)
                    zone_transfers.append(ZoneTransferItem(
                        device=machine_name,
                        zone="global",
                        allowed_hosts=allowed_hosts
                    ))
        
        return zone_transfers
    
    def _parse_allow_transfer(self, line: str) -> List[str]:
        """Parse allow-transfer directive to extract allowed hosts."""
        # Find the content between { and }
        start = line.find('{')
        end = line.rfind('}')
        
        if start == -1 or end == -1:
            return []
        
        content = line[start+1:end].strip()
        
        # Handle special cases
        if 'any' in content.lower():
            return ["any"]
        if 'none' in content.lower():
            return ["none"]
        
        # Split by semicolon and clean up
        hosts = []
        for host in content.split(';'):
            host = host.strip().strip('"').strip("'")
            if host and host != ',':
                hosts.append(host)
        
        return hosts
