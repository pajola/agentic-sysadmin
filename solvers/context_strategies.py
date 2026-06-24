"""
Deterministic context assembly and network probing strategies.

Each strategy gathers exactly the data needed for a category of questions,
without any LLM calls.  The assembled context is a formatted string of
file contents; probe results are raw command outputs.

Strategies:
  - topology_only:      lab.conf only
  - ip_analysis:        lab.conf + all .startup files
  - device_pair:        lab.conf + .startup files for two specific devices
  - live_connectivity:  files + traceroute/ping on the live network
  - service_scan:       all files + ps aux / DNS config on the live network
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: str) -> Optional[str]:
    """Read a file, returning None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


def _format_context(files: Dict[str, str]) -> str:
    """Format a dict of {display_path: content} into the bulk-style context string."""
    parts = []
    for display_path, content in sorted(files.items()):
        parts.append(f"**PATH:** {display_path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def _lab_base(lab_path: str) -> str:
    """Extract the lab directory base name."""
    return os.path.basename(os.path.normpath(lab_path))


def _read_lab_conf(lab_path: str) -> Dict[str, str]:
    """Read lab.conf and return as {display_path: content}."""
    base = _lab_base(lab_path)
    conf_path = os.path.join(lab_path, "lab.conf")
    content = _read_file(conf_path)
    if content is not None:
        return {f"{base}/lab.conf": content}
    return {}


def _read_startup_files(lab_path: str, device_filter: Optional[List[str]] = None) -> Dict[str, str]:
    """Read .startup files, optionally filtered to specific device names."""
    base = _lab_base(lab_path)
    files = {}
    for fname in sorted(os.listdir(lab_path)):
        if not fname.endswith(".startup"):
            continue
        device_name = fname[:-8]  # strip ".startup"
        if device_filter is not None and device_name not in device_filter:
            continue
        content = _read_file(os.path.join(lab_path, fname))
        if content is not None:
            files[f"{base}/{fname}"] = content
    return files


def _read_all_config_files(lab_path: str) -> Dict[str, str]:
    """Read ALL files in the lab directory (startup, conf, and nested configs)."""
    base = _lab_base(lab_path)
    files = {}
    for root, _dirs, fnames in os.walk(lab_path):
        for fname in fnames:
            abs_path = os.path.join(root, fname)
            rel = os.path.relpath(abs_path, lab_path).replace(os.sep, "/")
            content = _read_file(abs_path)
            if content is not None:
                files[f"{base}/{rel}"] = content
    return files


def _get_device_names(lab_path: str) -> List[str]:
    """List device names from .startup files."""
    devices = []
    for fname in os.listdir(lab_path):
        if fname.endswith(".startup"):
            devices.append(fname[:-8])
    return sorted(devices)


def _collect_runtime_ips(question, device_filter: Optional[List[str]] = None) -> str:
    """Run `ip -br addr show` on each device via Kathara and format the result.

    Returns an empty string if no Kathara client is available. This is the same
    data source as the ground truth (`ip a`), so it captures runtime-only
    addresses (IPv6 link-local, loopback, DHCP/SLAAC) that static .startup
    files do not contain.
    """
    kathara = getattr(question, "_kathara", None)
    if kathara is None:
        return ""

    try:
        machines_ips = kathara.get_machines_ips()
    except Exception as e:
        logger.warning(f"Could not list machines from Kathara: {e}")
        return ""

    devices = sorted(machines_ips.keys())
    if device_filter is not None:
        devices = [d for d in devices if d in device_filter]

    parts = []
    for device in devices:
        try:
            output = kathara.execute_command(device, "ip -br addr show")
        except Exception as e:
            logger.warning(f"ip addr show failed on {device}: {e}")
            output = None
        if output:
            parts.append(f"**RUNTIME IPs on {device}:**\n```\n{output.rstrip()}\n```")

    if not parts:
        return ""
    return "## Runtime IP configuration (from `ip -br addr show`)\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Context Assembly Strategies
# ---------------------------------------------------------------------------

def assemble_topology_only(lab_path: str, **kwargs) -> str:
    """Gather only lab.conf."""
    files = _read_lab_conf(lab_path)
    return _format_context(files)


def assemble_ip_analysis(lab_path: str, question=None, **kwargs) -> str:
    """Gather lab.conf + all .startup files + runtime IPs (if Kathara available)."""
    files = _read_lab_conf(lab_path)
    files.update(_read_startup_files(lab_path))
    static_ctx = _format_context(files)

    runtime_ctx = _collect_runtime_ips(question) if question is not None else ""
    if runtime_ctx:
        return f"{static_ctx}\n\n{runtime_ctx}"
    return static_ctx


def assemble_device_pair(lab_path: str, question=None, **kwargs) -> str:
    """Gather lab.conf + .startup files for the two named devices + their runtime IPs."""
    # Extract device names from the question object
    m1 = getattr(question, "m1", None) or getattr(question, "_device1", None)
    m2 = getattr(question, "m2", None) or getattr(question, "_device2", None)

    files = _read_lab_conf(lab_path)

    if m1 and m2:
        files.update(_read_startup_files(lab_path, device_filter=[m1, m2]))
    else:
        # Fallback: read all startup files if we can't determine the pair
        logger.warning("device_pair strategy: could not extract m1/m2 from question, reading all startup files")
        files.update(_read_startup_files(lab_path))

    static_ctx = _format_context(files)

    runtime_ctx = ""
    if question is not None and m1 and m2:
        runtime_ctx = _collect_runtime_ips(question, device_filter=[m1, m2])
    if runtime_ctx:
        return f"{static_ctx}\n\n{runtime_ctx}"
    return static_ctx


def assemble_live_connectivity(lab_path: str, question=None, **kwargs) -> str:
    """Gather lab.conf + all .startup files (needed for IP-to-device mapping)."""
    files = _read_lab_conf(lab_path)
    files.update(_read_startup_files(lab_path))
    return _format_context(files)


def assemble_service_scan(lab_path: str, **kwargs) -> str:
    """Gather ALL files (lab.conf, .startup, nested configs like named.conf)."""
    files = _read_all_config_files(lab_path)
    return _format_context(files)


# Strategy registry: strategy_name -> assembler function
CONTEXT_ASSEMBLERS = {
    "topology_only": assemble_topology_only,
    "ip_analysis": assemble_ip_analysis,
    "device_pair": assemble_device_pair,
    "live_connectivity": assemble_live_connectivity,
    "service_scan": assemble_service_scan,
}


# ---------------------------------------------------------------------------
# Network Probing Strategies
# ---------------------------------------------------------------------------

def probe_live_connectivity(question, lab_path: str) -> str:
    """Run traceroute from m1 to m2 using the live Kathara lab.

    Returns a formatted string with the traceroute output and device IP info.
    """
    kathara = getattr(question, "_kathara", None)
    if kathara is None:
        return "[ERROR] No Kathara client available — cannot probe live network."

    m1 = getattr(question, "m1", None)
    m2 = getattr(question, "m2", None)
    if not m1 or not m2:
        return "[ERROR] Could not determine source/destination devices from question."

    results_parts = []

    # Get IPs of the target device to try traceroute
    machines_ips = kathara.get_machines_ips()
    m2_ips = machines_ips.get(m2, set())

    if not m2_ips:
        return f"[ERROR] No IPs found for device '{m2}'."

    # Try traceroute to each non-special IP of m2
    for ip, cidr, is_special in m2_ips:
        if is_special:
            continue
        output, error = kathara.execute_command(m1, f"traceroute -n -w 2 {ip}", return_error=True)
        results_parts.append(
            f"## Traceroute from {m1} to {m2} (IP: {ip})\n```\n{output or 'No output'}\n```"
        )
        if error and error.strip():
            results_parts.append(f"stderr: {error.strip()}")

    # Also provide the IP-to-device mapping for interpretation
    results_parts.append("\n## Device IP Mapping")
    for device, ips in sorted(machines_ips.items()):
        ip_list = [f"{ip}/{cidr}" for ip, cidr, is_special in ips if not is_special]
        if ip_list:
            results_parts.append(f"- **{device}**: {', '.join(ip_list)}")

    return "\n\n".join(results_parts)


def probe_service_scan(question, lab_path: str) -> str:
    """Run ps aux on all devices and read DNS configs where applicable.

    Returns a formatted string with process listings and DNS configurations.
    """
    kathara = getattr(question, "_kathara", None)
    if kathara is None:
        return "[ERROR] No Kathara client available — cannot probe live network."

    machines_ips = kathara.get_machines_ips()
    results_parts = []

    # Phase 1: Run ps aux on every device
    dns_devices = []
    for device in sorted(machines_ips.keys()):
        output = kathara.execute_command(device, "ps aux")
        if output:
            results_parts.append(f"## Processes on {device}\n```\n{output}\n```")
            # Detect DNS servers for phase 2
            if any(name in output for name in ("named", "bind9", "dnsmasq", "unbound")):
                dns_devices.append(device)
        else:
            results_parts.append(f"## Processes on {device}\n(no output)")

    # Phase 2: Read DNS configuration files from detected DNS servers
    if dns_devices:
        results_parts.append("\n## DNS Configuration Files")
        dns_config_paths = [
            "/etc/bind/named.conf",
            "/etc/bind/named.conf.local",
            "/etc/bind/named.conf.default-zones",
            "/etc/named.conf",
        ]
        for device in dns_devices:
            for config_path in dns_config_paths:
                output, error = kathara.execute_command(
                    device, f"cat {config_path}", return_error=True
                )
                if output and output.strip():
                    results_parts.append(
                        f"### {device}: {config_path}\n```\n{output}\n```"
                    )

    return "\n\n".join(results_parts)


def probe_noop(question, lab_path: str) -> str:
    """No probing needed for static strategies."""
    return ""


# Strategy registry: strategy_name -> prober function
PROBERS = {
    "topology_only": probe_noop,
    "ip_analysis": probe_noop,
    "device_pair": probe_noop,
    "live_connectivity": probe_live_connectivity,
    "service_scan": probe_service_scan,
}
