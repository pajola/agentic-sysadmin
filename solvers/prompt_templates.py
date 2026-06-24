"""
Prompt templates for the StrategicAgentSolver.

Contains:
  - CLASSIFIER_PROMPT: used by the strategy_classifier node to choose a retrieval strategy
  - ANALYST_PROMPTS: per-strategy prompt templates for the analyst node
"""

# ---------------------------------------------------------------------------
# Strategy Classifier Prompt
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """You are classifying a network analysis question to determine the best data retrieval strategy for a Kathara virtual network lab.

Available strategies:

- **topology_only**: Only read the lab.conf topology file (device-to-network mappings). Use when the question is about network structure, device count, or connectivity layout — NOT about IP addresses or services.

- **ip_analysis**: Read lab.conf + all device startup scripts (.startup files containing IP configuration commands). Use when the question involves IP addresses, subnets, CIDR notation, IPv6, special-purpose IPs, or network addressing.

- **device_pair**: Read lab.conf + startup scripts of two specific devices mentioned in the question. Use when the question is about the relationship between exactly two named devices (shared subnet, direct link, comparison).

- **live_connectivity**: Read configuration files + execute live network commands (ping, traceroute) on the running lab. Use when the question requires testing actual network reachability, path tracing, or runtime connectivity that cannot be determined from configuration files alone.

- **service_scan**: Read all configuration files + check running processes and service configurations on all devices. Use when the question is about running services, daemons, DNS configuration, zone transfers, or application-level settings.

Question: {user_query}

Expected output schema:
```json
{output_schema}
```

Choose the single most appropriate strategy."""


# ---------------------------------------------------------------------------
# Analyst Prompt Templates (per strategy)
# ---------------------------------------------------------------------------

ANALYST_PROMPTS = {
    "topology_only": """You are an expert network administrator analyzing a Kathara virtual network lab.

# Task
Answer the following question using ONLY the lab configuration provided below.

# How to read lab.conf
In lab.conf, each line like `device_name[N]=collision_domain` declares that device `device_name` has interface ethN connected to the collision domain (virtual LAN) named `collision_domain`. Count unique device names (the part before the bracket) to find the total number of devices.

# Lab Configuration
{assembled_context}

# Output Schema
Your answer must match this JSON schema:
```json
{output_schema}
```

# Question
{user_query}

Analyze the configuration carefully and provide your answer.""",

    "ip_analysis": """You are an expert network administrator analyzing a Kathara virtual network lab.

# Task
Answer the following question by analyzing the IP address configuration of the lab.

# How to read the data
- **lab.conf**: Declares devices and their interface-to-network mappings.
- **.startup files**: Contain shell commands run at device boot. Look for:
  - `ip address add X.X.X.X/CIDR dev ethN` — assigns an IP address to an interface
  - `ip addr add ...` — same command, shortened form
  - `ifconfig ethN X.X.X.X netmask ...` — legacy IP assignment
- **Runtime IP configuration** (output of `ip -br addr show`, when present): the authoritative live state of each device. Format is `IFACE STATE IP1/CIDR IP2/CIDR ...`. This is the SAME source the ground truth uses, so prefer it over the .startup files when both are available. It also contains addresses that .startup files do NOT show, in particular:
  - Loopback (`127.0.0.1/8`, `::1/128`) — present on every Linux device
  - IPv6 link-local (`fe80::xxxx/64`) — auto-configured by the kernel on every active interface
  - Any DHCP/SLAAC-assigned address

# IP Classification
- **Special-purpose IPs**: loopback (`127.x.x.x`, `::1`), link-local (`169.254.x.x`, `fe80::/10`), multicast (`224-239.x.x.x`, `ff00::/8`), reserved (`240+`), unspecified (`0.0.0.0`, `::`).
- **IPv6 addresses** contain colons (e.g., `2001:db8::1/64`, `fe80::5054:ff:fe12:3456/64`).
- To compute a **subnet**: apply the CIDR mask to the IP address to get the network address (e.g., `10.0.1.5/24` → `10.0.1.0/24`).

# Lab Configuration
{assembled_context}

# Output Schema
Your answer must match this JSON schema:
```json
{output_schema}
```

# Question
{user_query}

Analyze all startup files systematically and provide your answer.""",

    "device_pair": """You are an expert network administrator analyzing a Kathara virtual network lab.

# Task
Answer the following question about the relationship between two specific devices in the network.

# How to read the data
- **lab.conf**: Shows which devices connect to which collision domains (LANs). Two devices on the same collision domain are directly connected at layer 2.
- **.startup files**: Contain IP address assignments. Two devices are on the same subnet if their IP/CIDR pairs resolve to the same network address.
- **Runtime IP configuration** (output of `ip -br addr show`, when present): the live state of each device's interfaces. Prefer this over the .startup files when both are available — it shows every address actually configured, including ones not in .startup (DHCP, SLAAC, link-local).

# Lab Configuration
{assembled_context}

# Output Schema
Your answer must match this JSON schema:
```json
{output_schema}
```

# Question
{user_query}

Compare the configurations of both devices carefully and provide your answer.""",

    "live_connectivity": """You are an expert network administrator analyzing a Kathara virtual network lab.

# Task
Answer the following question about network connectivity. Live network probe results have been gathered for you.

# How to interpret the data
- **Traceroute output**: Each numbered line is a hop. If only 1 hop appears (the destination), the devices communicate directly. Multiple hops indicate intermediate routers. Lines with `* * *` indicate unreachable hops.
- **Ping output**: Look for "bytes from" lines indicating successful replies. "100% packet loss" means unreachable.
- **Device-to-IP mapping**: Use the startup files to map IP addresses to device names.

# Lab Configuration
{assembled_context}

# Network Probe Results
{probe_results}

# Output Schema
Your answer must match this JSON schema:
```json
{output_schema}
```

# Question
{user_query}

Analyze the probe results together with the configuration files and provide your answer.""",

    "service_scan": """You are an expert network administrator analyzing a Kathara virtual network lab.

# Task
Answer the following question about running services and their configurations.

# How to interpret the data
- **`ps aux` output**: Shows all running processes on each device. Look for these service types:
  - Web servers: apache2, nginx, httpd, lighttpd
  - FTP servers: proftpd, vsftpd, pure-ftpd
  - DNS servers: named, bind9, dnsmasq, unbound
  - Mail servers: postfix, sendmail, exim4, dovecot
  - Proxy servers: squid, haproxy
  - File sharing: smbd, nmbd, nfsd, netatalk
- **DNS configuration** (named.conf): Check `allow-transfer` directives inside zone blocks. `allow-transfer {{ any; }}` means the zone is vulnerable to zone transfer from any host. `allow-transfer {{ none; }}` or no directive means protected.
- Only report hosts that have at least one running service.

# Lab Configuration
{assembled_context}

# Service Probe Results
{probe_results}

# Output Schema
Your answer must match this JSON schema:
```json
{output_schema}
```

# Question
{user_query}

Analyze the process listings and configurations carefully and provide your answer.""",
}
