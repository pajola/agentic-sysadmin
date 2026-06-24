from typing import List, Type, Optional
from pydantic import BaseModel, Field, validator
from .base_question import BaseQuestion

class ServiceItem(BaseModel):
    service_type: str = Field(description="Type of service: web, ftp, dns, mail, proxy, or file_share")
    service_name: str = Field(description="Specific service name (e.g., apache2, bind9, proftpd)")

class HostServicesItem(BaseModel):
    device: str = Field(description="Device/host name")
    services: List[ServiceItem] = Field(description="List of application services (not routing services) running on this host", min_items=1)
    
    @validator('services')
    def services_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Services list cannot be empty - only include hosts with running services')
        return v

class EnabledServicesResponse(BaseModel):
    hosts: List[HostServicesItem] = Field(
        description="List of hosts with their running application services (web, ftp, dns, mail, proxy, file_share only). Only include hosts that have at least one running service."
    )

class EnabledServicesQuestion(BaseQuestion):
    """Question plugin to identify which services are running on each host."""

    def __init__(self, lab_whitelist: Optional[List[str]] = None):
        super().__init__(lab_whitelist=lab_whitelist)

    @property
    def question_text(self) -> str:
        return "Which application services are running on each host? Only report web servers (apache2, nginx), FTP servers (proftpd, vsftpd), DNS servers (named, bind9), mail servers (postfix, sendmail), proxy servers (squid, haproxy), and file sharing services (samba, nfs). Do not include hosts with no running services."

    @staticmethod
    def output_model() -> Type[BaseModel]:
        return EnabledServicesResponse

    def get_ground_truth(self) -> BaseModel:
        machines_ips = self._kathara.get_machines_ips()
        hosts_with_services = []
        
        # Simple process-based detection
        service_processes = {
            "web": ["apache2", "nginx", "httpd", "lighttpd"],
            "ftp": ["proftpd", "vsftpd", "pure-ftpd"],
            "dns": ["named", "bind9", "dnsmasq", "unbound"],
            "mail": ["postfix", "sendmail", "exim4", "dovecot"],
            "proxy": ["squid", "haproxy"],
            "file_share": ["smbd", "nmbd", "nfsd", "netatalk"]
        }
        
        for machine_name in machines_ips.keys():
            # Get all running processes - single command per machine
            process_output = self._kathara.execute_command(machine_name, "ps aux")
            if not process_output:
                continue
                
            machine_services = []
            
            for service_type, processes in service_processes.items():
                for process_name in processes:
                    if process_name in process_output:
                        machine_services.append(ServiceItem(
                            service_type=service_type,
                            service_name=process_name
                        ))
                        break  # Only count one instance per service type per host
            
            if machine_services:
                hosts_with_services.append(HostServicesItem(
                    device=machine_name,
                    services=machine_services
                ))
        
        return EnabledServicesResponse(
            hosts=hosts_with_services
        )
