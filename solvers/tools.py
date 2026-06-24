"""
Shared tool factories for agentic solvers.

Each factory takes the required dependencies (lab_path, kathara_client) and returns
a @tool-decorated function ready to be bound to a LangChain model via .bind_tools().

Tools are grouped into two categories:
  - File tools: read lab configuration files (no running lab needed)
  - Network tools: execute commands on a live Kathara lab
"""

import os
import logging
from typing import Dict, List, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-based tools — work on the static lab directory
# ---------------------------------------------------------------------------

def make_list_lab_files(lab_path: str):
    """Factory: returns a tool that lists every file in the lab directory."""

    @tool
    def list_lab_files() -> Dict[str, Any]:
        """List all files in the Kathara lab directory, organized by category.

        Use this tool first to understand what files are available before reading
        specific ones.  Files are grouped into:
          - topology: lab.conf and similar topology definition files
          - startup:  <device>.startup scripts executed at device boot
          - config:   extra configuration files inside device subdirectories
                      (e.g. router1/etc/bind/named.conf)

        All paths are returned **relative to the lab root**, so you can pass
        them directly to `read_file(relative_path=...)`.  Examples:
            "lab.conf"
            "router1.startup"
            "router1/etc/bind/named.conf"

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - topology (list[str]): topology file paths relative to lab root
              - startup  (list[str]): startup file paths relative to lab root
              - config   (list[str]): other config file paths relative to lab root
        """
        try:
            topology_files: List[str] = []
            startup_files: List[str] = []
            config_files: List[str] = []

            for root, _dirs, files in os.walk(lab_path):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    rel = os.path.relpath(abs_path, lab_path).replace(os.sep, "/")

                    if rel == "lab.conf" or rel.endswith(".conf") and "/" not in rel:
                        topology_files.append(rel)
                    elif rel.endswith(".startup"):
                        startup_files.append(rel)
                    else:
                        config_files.append(rel)

            return {
                "status": "success",
                "topology": sorted(topology_files),
                "startup": sorted(startup_files),
                "config": sorted(config_files),
            }
        except Exception as e:
            logger.error(f"Error listing lab files in {lab_path}: {e}")
            return {"status": "error", "topology": [], "startup": [], "config": []}

    return list_lab_files


def make_read_lab_conf(lab_path: str):
    """Factory: returns a tool that reads the lab.conf topology file."""

    @tool
    def read_lab_conf() -> Dict[str, Any]:
        """Read the lab.conf file that defines the network topology.

        lab.conf declares:
          - Which devices (routers, hosts, servers) exist
          - Which network interface of each device connects to which
            collision domain (virtual LAN segment)

        Syntax example:
          router1[0]=net_A       → router1 eth0 is on collision domain net_A
          router1[1]=net_B       → router1 eth1 is on collision domain net_B
          server1[0]=net_A       → server1 eth0 is on collision domain net_A

        Two devices on the same collision domain can communicate at layer 2.

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - content (str): the full text of lab.conf
              - message (str): human-readable note (on error)
        """
        conf_path = os.path.join(lab_path, "lab.conf")
        try:
            with open(conf_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "content": content}
        except FileNotFoundError:
            return {
                "status": "error",
                "content": "",
                "message": "lab.conf not found in the lab directory.",
            }
        except Exception as e:
            logger.error(f"Error reading lab.conf at {conf_path}: {e}")
            return {"status": "error", "content": "", "message": str(e)}

    return read_lab_conf


def make_read_file(lab_path: str):
    """Factory: returns a tool that reads any file by its relative path."""

    @tool
    def read_file(relative_path: str) -> Dict[str, Any]:
        """Read a specific file from the lab directory by its relative path.

        Use this when you need to read a single file precisely — for example
        a specific device's startup script or a nested config file.

        The relative_path should be relative to the lab root directory,
        e.g. "router1.startup" or "router1/etc/bind/named.conf".

        You can discover available paths with the list_lab_files tool.

        Args:
            relative_path: path relative to the lab root
                           (e.g. "router1.startup", "lab.conf",
                            "dns_server/etc/bind/named.conf")

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - path (str): the requested relative path
              - content (str): file contents (empty on error)
              - message (str): explanation on error
        """
        # Prevent path traversal
        safe = os.path.normpath(relative_path)
        if safe.startswith("..") or os.path.isabs(safe):
            return {
                "status": "error",
                "path": relative_path,
                "content": "",
                "message": "Invalid path: must be relative and inside the lab directory.",
            }

        full_path = os.path.join(lab_path, safe)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "path": relative_path, "content": content}
        except FileNotFoundError:
            return {
                "status": "error",
                "path": relative_path,
                "content": "",
                "message": f"File not found: {relative_path}. Use list_lab_files() to see available files.",
            }
        except Exception as e:
            logger.error(f"Error reading file {full_path}: {e}")
            return {
                "status": "error",
                "path": relative_path,
                "content": "",
                "message": str(e),
            }

    return read_file


def make_get_devices_name(lab_path: str):
    """Factory: returns a tool that lists all device names in the lab."""

    @tool
    def get_devices_name() -> Dict[str, Any]:
        """Get the names of all network devices defined in the lab.

        Device names are derived from .startup files: a file named
        "router1.startup" means there is a device called "router1".

        This is useful to know which devices exist before fetching their
        individual configurations with get_device_config.

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - devices (list[str]): list of device name strings
        """
        devices: List[str] = []
        try:
            for filename in os.listdir(lab_path):
                if filename.endswith(".startup"):
                    devices.append(filename[:-8])  # strip ".startup"
            return {"status": "success", "devices": sorted(devices)}
        except Exception as e:
            logger.error(f"Error reading devices from {lab_path}: {e}")
            return {"status": "error", "devices": []}

    return get_devices_name


def make_get_device_config(lab_path: str):
    """Factory: returns a tool that reads ALL config files for a given device."""

    @tool
    def get_device_config(device_name: str) -> Dict[str, Any]:
        """Retrieve every configuration file associated with a device.

        This reads:
          1. Files in the lab root starting with "<device_name>."
             (e.g. router1.startup)
          2. All files inside the "<device_name>/" subdirectory tree
             (e.g. router1/etc/quagga/bgpd.conf)

        Use this when you need the complete picture of a device's config.
        If you only need a single file, prefer read_file() for efficiency.

        Args:
            device_name: exact device name (e.g. "router1", "dns_server").
                         Use get_devices_name() first if unsure of the name.

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - device_name (str): the queried device name
              - device_config (dict[str, str]): mapping of relative file paths
                to their contents. Empty dict if device not found.
        """
        try:
            device_files: Dict[str, str] = {}

            # Files starting with device_name.
            for filename in os.listdir(lab_path):
                if filename.startswith(device_name + "."):
                    file_path = os.path.join(lab_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                device_files[filename] = f.read()
                        except Exception as e:
                            logger.error(f"Error reading {filename}: {e}")
                            device_files[filename] = f"Error reading file: {e}"

            # Files inside device_name/ subdirectory
            device_folder = os.path.join(lab_path, device_name)
            if os.path.isdir(device_folder):
                for root, _, files in os.walk(device_folder):
                    for fname in files:
                        abs_path = os.path.join(root, fname)
                        rel = os.path.relpath(abs_path, lab_path).replace(os.sep, "/")
                        try:
                            with open(abs_path, "r", encoding="utf-8") as f:
                                device_files[rel] = f.read()
                        except Exception as e:
                            logger.error(f"Error reading {rel}: {e}")
                            device_files[rel] = f"Error reading file: {e}"

            if not device_files:
                return {
                    "status": "error",
                    "device_name": device_name,
                    "device_config": f"Device '{device_name}' not found. "
                    "Use get_devices_name() to list available devices.",
                }

            return {
                "status": "success",
                "device_name": device_name,
                "device_config": device_files,
            }
        except Exception as e:
            logger.error(f"Error getting config for {device_name}: {e}")
            return {
                "status": "error",
                "device_name": device_name,
                "device_config": f"Error: {e}",
            }

    return get_device_config


# ---------------------------------------------------------------------------
# Network tools — require a running Kathara lab (KatharaClient)
# ---------------------------------------------------------------------------

def make_get_running_processes(kathara_client):
    """Factory: returns a tool that lists all running processes on a device."""

    @tool
    def get_running_processes(device_name: str) -> Dict[str, Any]:
        """List all running processes on a network device.

        Runs 'ps aux' inside the device's container to show every active
        process.  Useful for identifying which application services are
        running on a device (e.g. apache2, named/bind9, proftpd, vsftpd,
        squid, haproxy, postfix, sendmail, smbd, nfsd).

        Args:
            device_name: name of the device (e.g. "router1", "dns_server").
                         Must match a running device in the lab.

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - device_name (str): queried device
              - output (str): raw 'ps aux' output listing all processes
              - error (str|None): stderr if any
        """
        output, error = kathara_client.execute_command(
            device_name, "ps aux", return_error=True
        )
        status = "error" if (error and error.strip()) else "success"
        return {
            "status": status,
            "device_name": device_name,
            "output": output or "",
            "error": error,
        }

    return get_running_processes


def make_read_device_file(kathara_client):
    """Factory: returns a tool that reads a file from a running device's filesystem."""

    @tool
    def read_device_file(device_name: str, file_path: str) -> Dict[str, Any]:
        """Read a file from the runtime filesystem of a running network device.

        Runs 'cat <file_path>' inside the device's container to retrieve
        the file contents.  This reads the live filesystem as seen by the
        device at runtime, which may differ from the static lab directory.

        Useful for reading configuration files that are only present inside
        the container (e.g. /etc/bind/named.conf, /etc/bind/named.conf.local,
        /etc/resolv.conf, /etc/apache2/apache2.conf).

        Args:
            device_name: name of the device (e.g. "dns_server", "router1").
                         Must match a running device in the lab.
            file_path: absolute path of the file inside the device
                       (e.g. "/etc/bind/named.conf", "/etc/resolv.conf")

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - device_name (str): queried device
              - file_path (str): the requested path
              - content (str): file contents (empty on error)
              - error (str|None): stderr if any
        """
        output, error = kathara_client.execute_command(
            device_name, f"cat {file_path}", return_error=True
        )
        status = "error" if (error and error.strip()) else "success"
        return {
            "status": status,
            "device_name": device_name,
            "file_path": file_path,
            "content": output or "",
            "error": error,
        }

    return read_device_file


def make_ping(kathara_client):
    """Factory: returns a tool that tests ICMP reachability between two devices."""

    @tool
    def ping(source_device: str, destination_ip: str, count: int = 3) -> Dict[str, Any]:
        """Test network reachability from one device to an IP address using ICMP ping.

        Sends ICMP echo requests from the source device to the destination.
        Useful to verify layer-3 connectivity and routing between devices.

        Args:
            source_device: name of the device to ping FROM (e.g. "router1")
            destination_ip: IP address or hostname to ping
                            (e.g. "10.0.1.1", "server1")
            count: number of ICMP packets to send (default 3)

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - source (str): source device name
              - destination (str): destination that was pinged
              - reachable (bool): True if at least one reply was received
              - output (str): raw ping output for inspection
        """
        cmd = f"ping -c {count} -W 2 {destination_ip}"
        output, error = kathara_client.execute_command(
            source_device, cmd, return_error=True
        )
        # ping returns exit code 1 on failure which may appear as error
        raw = output or ""
        reachable = "bytes from" in raw.lower() or "ttl=" in raw.lower()
        return {
            "status": "success",
            "source": source_device,
            "destination": destination_ip,
            "reachable": reachable,
            "output": raw,
        }

    return ping


def make_traceroute(kathara_client):
    """Factory: returns a tool that traces the network path between two devices."""

    @tool
    def traceroute(source_device: str, destination_ip: str) -> Dict[str, Any]:
        """Trace the network path (layer-3 hops) from a device to a destination.

        Runs traceroute inside the source device's container.  Each hop in the
        output represents a router or gateway between source and destination.

        Useful for understanding routing paths and diagnosing connectivity issues.

        Args:
            source_device: name of the device to traceroute FROM (e.g. "router1")
            destination_ip: IP address or hostname of the destination
                            (e.g. "10.0.1.1", "server1")

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - source (str): source device name
              - destination (str): destination address
              - output (str): raw traceroute output
              - error (str|None): stderr if any
        """
        cmd = f"traceroute -n -w 2 {destination_ip}"
        output, error = kathara_client.execute_command(
            source_device, cmd, return_error=True
        )
        status = "error" if (error and error.strip()) else "success"
        return {
            "status": status,
            "source": source_device,
            "destination": destination_ip,
            "output": output or "",
            "error": error,
        }

    return traceroute


def make_get_routing_table(kathara_client):
    """Factory: returns a tool that retrieves a device's IP routing table."""

    @tool
    def get_routing_table(device_name: str) -> Dict[str, Any]:
        """Get the full IP routing table of a network device.

        Runs 'ip route show' on the device to display all routes including
        directly connected networks, static routes, and dynamically learned
        routes (OSPF, BGP, etc.).

        Each line shows: destination network, gateway, interface, and metric.

        Args:
            device_name: name of the device (e.g. "router1")

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - device_name (str): queried device
              - output (str): raw 'ip route show' output
              - error (str|None): stderr if any
        """
        output, error = kathara_client.execute_command(
            device_name, "ip route show", return_error=True
        )
        status = "error" if (error and error.strip()) else "success"
        return {
            "status": status,
            "device_name": device_name,
            "output": output or "",
            "error": error,
        }

    return get_routing_table


def make_get_interfaces(kathara_client):
    """Factory: returns a tool that retrieves a device's network interfaces and IPs."""

    @tool
    def get_interfaces(device_name: str) -> Dict[str, Any]:
        """Get all network interfaces and their IP addresses for a device.

        Runs 'ip addr show' on the device.  The output lists every interface
        (lo, eth0, eth1, …) with its state (UP/DOWN), MAC address, and all
        assigned IPv4/IPv6 addresses with prefix length.

        Args:
            device_name: name of the device (e.g. "router1")

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - device_name (str): queried device
              - output (str): raw 'ip addr show' output
              - error (str|None): stderr if any
        """
        output, error = kathara_client.execute_command(
            device_name, "ip addr show", return_error=True
        )
        status = "error" if (error and error.strip()) else "success"
        return {
            "status": status,
            "device_name": device_name,
            "output": output or "",
            "error": error,
        }

    return get_interfaces


def make_get_arp_table(kathara_client):
    """Factory: returns a tool that retrieves a device's ARP table."""

    @tool
    def get_arp_table(device_name: str) -> Dict[str, Any]:
        """Get the ARP (Address Resolution Protocol) table of a device.

        Shows the mapping between IP addresses and MAC addresses that the
        device has learned.  Useful for verifying layer-2 adjacency and
        identifying which devices are directly reachable.

        Args:
            device_name: name of the device (e.g. "router1")

        Returns:
            dict with keys:
              - status (str): "success" or "error"
              - device_name (str): queried device
              - output (str): raw 'arp -n' output
              - error (str|None): stderr if any
        """
        output, error = kathara_client.execute_command(
            device_name, "arp -n", return_error=True
        )
        status = "error" if (error and error.strip()) else "success"
        return {
            "status": status,
            "device_name": device_name,
            "output": output or "",
            "error": error,
        }

    return get_arp_table


# ---------------------------------------------------------------------------
# Helper: build the tool list for a solver
# ---------------------------------------------------------------------------

def build_file_tools(lab_path: str) -> List:
    """Create all file-based tools for a given lab path.

    Returns:
        List of tool functions: [list_lab_files, read_lab_conf, read_file,
                                  get_devices_name, get_device_config]
    """
    return [
        make_list_lab_files(lab_path),
        make_read_lab_conf(lab_path),
        make_read_file(lab_path),
        make_get_devices_name(lab_path),
        make_get_device_config(lab_path),
    ]


def build_network_tools(kathara_client) -> List:
    """Create all network-based tools for a given KatharaClient.

    Returns:
        List of tool functions: [ping, traceroute, get_routing_table,
                                  get_interfaces, get_arp_table,
                                  get_running_processes, read_device_file]
    """
    return [
        make_ping(kathara_client),
        make_traceroute(kathara_client),
        make_get_routing_table(kathara_client),
        make_get_interfaces(kathara_client),
        make_get_arp_table(kathara_client),
        make_get_running_processes(kathara_client),
        make_read_device_file(kathara_client),
    ]
