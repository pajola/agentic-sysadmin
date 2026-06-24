import atexit
import logging
import re
import time
import random
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from functools import wraps
from Kathara.manager.Kathara import Kathara
from Kathara.parser.netkit.LabParser import LabParser
from Kathara.model.Lab import Lab
import ipaddress

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def network_stable_retry(max_retries: int = 5, consistency_checks: int = 2, delay: float = 2.0):
    """
    Decorator that retries network operations until consistent results are obtained.
    
    Args:
        max_retries: Maximum number of retry attempts
        consistency_checks: Number of consecutive identical results needed
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            results_history = []
            
            for attempt in range(max_retries):
                try:
                    result = func(self, *args, **kwargs)
                    results_history.append(result)
                    
                    # Check if we have enough consecutive identical results
                    if len(results_history) >= consistency_checks:
                        recent_results = results_history[-consistency_checks:]
                        if all(r == recent_results[0] for r in recent_results):
                            logger.debug(f"{func.__name__} converged after {attempt + 1} attempts")
                            return result

                    
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        time.sleep(delay)
                        
                except Exception as e:
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            
            # Return the last result if we never got consistency
            logger.warning(f"{func.__name__} did not converge, returning last result")
            return results_history[-1] if results_history else None
            
        return wrapper
    return decorator

class KatharaClient:
    """
    Provides an interface to interact with a Kathara lab instance.
    All Kathara-specific operations are encapsulated here.
    """
    
    # Track the currently active client to avoid multiple labs being active
    _active_client: Optional["KatharaClient"] = None

    def __init__(self, lab: Lab):
        """
        Initialize with a Kathara lab instance.
        
        Args:
            lab (Lab): The Kathara lab instance to work with
        """
        self._lab = lab
        # Cache for IP addresses to avoid repeated computation
        self._machines_ips: Optional[Dict[str, Set[Tuple[str, str, bool]]]] = None
        # Caches for forwarding-plane validation (traceroute grading).
        self._routing_tables: Dict[str, list] = {}
        self._ip_owner: Optional[Dict[str, str]] = None
        atexit.register(self.undeploy)

    # Context manager support to ensure cleanup
    def __enter__(self) -> "KatharaClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.undeploy()

    def undeploy(self) -> None:
        """
        Stop and remove the Kathara lab instance.
        """
        if self._lab:
            try:
                Kathara.get_instance().undeploy_lab(lab=self._lab)
                print("Kathara lab successfully undeployed")
            except Exception as e:
                print(f"Error while undeploying Kathara lab: {e}")
            finally:
                self._lab = None
                self._machines_ips = None
                self._routing_tables = {}
                self._ip_owner = None

    @staticmethod
    def wipe() -> None:
        """
        Stop and remove all Kathara labs and clean up the environment globally.
        Equivalent to the 'kathara wipe' CLI command.
        Uses the internal Kathara manager to safely remove containers and networks.
        """
        try:
            print("Wiping all Kathara resources (may take a few seconds)...")
            Kathara.get_instance().wipe()
            print("Kathara environment successfully wiped")
            
            # Clear active client tracking if any
            if KatharaClient._active_client:
                KatharaClient._active_client._lab = None
                KatharaClient._active_client._machines_ips = None
                KatharaClient._active_client = None
        except Exception as e:
            print(f"Error while wiping Kathara: {e}")
            print("Consider manual cleanup: 'docker rm -f $(docker ps -aq) && docker network prune -f'")

    @classmethod
    def from_lab_path(cls, lab_path: str) -> 'KatharaClient':
        """
        Create a KatharaClient instance from a lab path.
        
        Args:
            lab_path (str): Path to the Kathara lab configuration
            
        Returns:
            KatharaClient: New instance initialized with the lab
        """
        # Always undeploy any previously active lab to avoid conflicts
        if cls._active_client is not None:
            try:
                cls._active_client.undeploy()
            except Exception as e:
                logger.warning(f"Failed to undeploy previous lab before deploying new one: {e}")

        # Logic to load lab from path
        lab = LabParser.parse(lab_path, "lab.conf")
        
        try:
            Kathara.get_instance().deploy_lab(lab)
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower() or "inconsistent state" in error_msg.lower():
                logger.error(f"Failed to deploy lab due to conflict: {error_msg}")
                logger.info("Attempting global Kathara wipe and retry...")
                # Automatic recovery attempt
                KATHARA_WIPE_CALLED = False
                try:
                    KatharaClient.wipe()
                    KATHARA_WIPE_CALLED = True
                    # Re-parse and re-deploy after wipe
                    lab = LabParser.parse(lab_path, "lab.conf")
                    Kathara.get_instance().deploy_lab(lab)
                    logger.info("Recovery successful: lab deployed after wipe.")
                except Exception as retry_e:
                    logger.error(f"Fails to recover after wipe: {retry_e}")
                    raise RuntimeError(
                        f"Kathara deployment failed: {error_msg}. "
                        "Automatic wipe attempt failed. Please perform manual cleanup: "
                        "docker rm -f $(docker ps -aq) && docker network prune -f"
                    ) from retry_e
            else:
                raise e

        instance = cls(lab)
        # Register as the active client
        KatharaClient._active_client = instance
        
        # Additional convergence check for complex networks
        if len(lab.machines) > 5:
            logger.info("Large network detected, performing convergence check...")
            instance._wait_for_convergence()

        return instance

    def _wait_for_convergence(self, timeout: int = 120) -> bool:
        """
        Perform a convergence check for complex networks.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if convergence detected, False if timeout
        """

        start_time = time.time()
        machines = list(self._lab.machines.keys())
        
        if len(machines) < 2:
            return True
        
        # Calculate test pairs based on network size
        test_pairs = self._select_convergence_test_pairs(machines)
        
        logger.debug(f"Testing convergence with {len(test_pairs)} machine pairs from {len(machines)} total machines")
        
        while time.time() - start_time < timeout:
            try:
                stable_count = 0
                for m1, m2 in test_pairs:
                    # Quick ping test to see if routing is working
                    result1 = self._quick_connectivity_test(m1, m2)
                    time.sleep(1)
                    result2 = self._quick_connectivity_test(m1, m2)
                    
                    if result1 == result2:  # Consistent results
                        stable_count += 1
                
                if stable_count == len(test_pairs):
                    logger.info("Network convergence detected")
                    return True
                    
            except Exception as e:
                logger.debug(f"Convergence check iteration failed: {e}")
            
            time.sleep(2)
        
        logger.warning("Network convergence check timeout - proceeding anyway")
        return False
    
    def _select_convergence_test_pairs(self, machines: List[str]) -> List[Tuple[str, str]]:
        """
        Select machine pairs for convergence testing based on network size.
        Uses a percentage-based approach with randomization for better coverage.
        
        Args:
            machines: List of machine names in the lab
            
        Returns:
            List of (machine1, machine2) tuples to test
        """
        num_machines = len(machines)
        
        # Calculate number of pairs to test based on network size
        if num_machines <= 3:
            # Small networks: test all possible pairs
            test_pairs = [(machines[i], machines[j]) 
                         for i in range(num_machines) 
                         for j in range(i + 1, num_machines)]
        elif num_machines <= 8:
            # Medium networks: test ~50% of possible pairs, minimum 3 pairs
            max_possible_pairs = (num_machines * (num_machines - 1)) // 2
            target_pairs = max(3, max_possible_pairs // 2)
            test_pairs = self._generate_random_pairs(machines, target_pairs)
        else:
            # Large networks: test ~25% but cap at 10 pairs for performance
            max_possible_pairs = (num_machines * (num_machines - 1)) // 2
            target_pairs = min(10, max(5, max_possible_pairs // 4))
            test_pairs = self._generate_random_pairs(machines, target_pairs)
        
        logger.debug(f"Selected {len(test_pairs)} test pairs from {num_machines} machines "
                    f"({len(test_pairs) / ((num_machines * (num_machines - 1)) // 2) * 100:.1f}% coverage)")
        
        return test_pairs
    
    def _generate_random_pairs(self, machines: List[str], target_count: int) -> List[Tuple[str, str]]:
        """
        Generate random unique machine pairs for testing.
        
        Args:
            machines: List of machine names
            target_count: Number of pairs to generate
            
        Returns:
            List of unique (machine1, machine2) tuples
        """
        # Generate all possible pairs
        all_pairs = [(machines[i], machines[j]) 
                    for i in range(len(machines)) 
                    for j in range(i + 1, len(machines))]
        
        # Shuffle and take target count
        random.shuffle(all_pairs)
        return all_pairs[:min(target_count, len(all_pairs))]
    
    def _quick_connectivity_test(self, m1: str, m2: str) -> Optional[bool]:
        """Quick ping test for convergence checking."""
        try:
            # Get first non-special IP of m2
            m2_ips = self.get_machines_ips().get(m2, set())
            for ip, cidr, is_special in m2_ips:
                if not is_special:
                    result = self.execute_command(m1, f"ping -c 1 -W 1 {ip}")
                    return "1 received" in result if result else False
            return None
        except Exception:
            return None

    def count_nodes(self) -> int:
        """
        Return the total number of nodes in the network using the deployed lab.
        """
        return len(self._lab.machines)

    def get_machines_ips(self) -> Dict[str, Set[Tuple[str, str, bool]]]:
        """
        Get IP addresses for all machines (with caching).
        
        Returns:
            Dict[str, Set[Tuple[str, str, bool]]]: Dictionary mapping machine names to sets of
            (ip_address, cidr, is_special) tuples, where:
                - ip_address (str): The IP address
                - cidr (str): The CIDR notation prefix length
                - is_special (bool): Whether this is a special-purpose IP (loopback, multicast, etc.)
        """
        if self._machines_ips is None:
            self._machines_ips = self._collect_machines_ips()
        return self._machines_ips
    
    def _collect_machines_ips(self) -> Dict[str, Set[Tuple[str, str, bool]]]:
        """
        Internal method to collect IP addresses from all machines.
        
        Returns:
            Dict[str, Set[Tuple[str, str, bool]]]: Dictionary mapping machine names to sets of
            (ip_address, cidr, is_special) tuples.
        """
        machines_ips = {}
        for machine in self._lab.machines.values():
            machines_ips[machine.name] = set()
            ips = self._get_ip_addresses(machine.name)
            for iface, ip, cidr in ips:
                is_special = False
                complete_ip = ipaddress.ip_address(ip)
                if (complete_ip.is_reserved or complete_ip.is_link_local or 
                    complete_ip.is_loopback or complete_ip.is_multicast or 
                    complete_ip.is_unspecified):
                    is_special = True
                machines_ips[machine.name].add((ip, cidr, is_special))
        return machines_ips
    
    def _get_ip_addresses(self, machine_name: str, ipv=None) -> List[Tuple[str, str, int]]:
        """
        Get IP addresses for all network interfaces.
        
        Args:
            machine_name (str): The name of the machine to query
            ipv (str): Can be 'ipv4', 'ipv6', or None (to get both).
        
        Returns:
            List of tuples: [(interface, ip_address, cidr), ...]
        """
        # Validate the argument
        if ipv not in {None, 'ipv4', 'ipv6'}:
            raise ValueError("Invalid argument: ipv must be 'ipv4', 'ipv6', or None.")
        
        try:
            # Run the `ip a` command to get all IP addresses
            output = self.execute_command(machine_name=machine_name, command=r"ip a")

        except FileNotFoundError:
            print("The 'ip' command is not available.")
            return []

        interfaces = []
        current_interface = None

        # Parse the output line by line
        for line in output.splitlines():
            line = line.strip()
            # Match the interface name (e.g., "1: lo:" or "2: eth0:")
            if re.match(r"^\d+:\s", line):
                # Extract the current interface name
                current_interface = line.split(":")[1].strip()
            # Match IPv4 or IPv6 addresses based on the `ipv` argument
            elif ipv in {None, 'ipv4'} and "inet " in line:  # IPv4
                ip_match = re.search(r"inet\s([\d\.]+/\d+)", line)
                if ip_match and current_interface:
                    ip_address, cidr = ip_match.group(1).split('/')
                    interfaces.append((current_interface, ip_address, int(cidr)))
            elif ipv in {None, 'ipv6'} and "inet6 " in line:  # IPv6
                ip_match = re.search(r"inet6\s([\d\w:]+/\d+)", line)
                if ip_match and current_interface:
                    ip_address, cidr = ip_match.group(1).split('/')
                    interfaces.append((current_interface, ip_address, int(cidr)))

        return interfaces

    def execute_command(self, machine_name: str, command: str, return_error: bool = False) -> Optional[str]:
        """
        Execute a command on a machine in the lab.
        
        Args:
            machine_name (str): Name of the machine
            command (str): Command to execute
            return_error (bool): If True, return a tuple (output, error). If False, return output only.
        Returns:
            Optional[str] or Tuple[Optional[str], Optional[str]]: Command output if successful, None otherwise. If return_error is True, returns (output, error).
        """
        try:
            result = Kathara.get_instance().exec(
                machine_name=machine_name,
                lab=self._lab,
                command=command,
                stream=False
            )
            output = result[0].decode('utf-8', errors='ignore') if result and result[0] else None
            error = result[1].decode('utf-8', errors='ignore') if result and result[1] else None
            if return_error:
                return (output, error)
            return output
        except Exception as e:
            if return_error:
                return (None, str(e))
            return None

    @network_stable_retry(max_retries=5, consistency_checks=2, delay=1.0)
    def traceroute(self, m1: str, m2: str) -> list[str]|None:
        """
        Execute a traceroute between two machines and parse the results.

        This method performs a network trace between two machines, handling various
        edge cases and protocol behaviors. It attempts to trace to each valid IP
        address of the destination machine until a successful path is found.

        Args:
            m1 (str): Source machine name
            m2 (str): Destination machine name

        Returns:
            list[str]|None: List of IP addresses in the path, or None if unreachable

        Raises:
            ValueError: If no IP addresses are found for the destination machine
        """
        machine_2_ips = self.get_machines_ips()[m2]
        if not machine_2_ips:
            raise ValueError(f"No IP addresses found for {m2}")

        for target_ip, cidr, is_special in machine_2_ips:

            network = ipaddress.ip_network(f"{target_ip}/{cidr}", strict=False)
            if (is_special):
                continue

            # Run the `traceroute` command on machine_1 to machine_2
            raw_output = Kathara.get_instance().exec(
                machine_name=m1, 
                lab=self._lab,  # Fixed: using self._lab instead of self.lab
                command=f"traceroute {target_ip}", 
                stream=False
            )
            if raw_output is None or raw_output[0] is None:
                continue
            output = raw_output[0].decode('utf-8')  # Decode the output from bytes to string

            hops_for_this_ip = []
            for line in output.splitlines():
                if line.startswith("traceroute"):
                    continue
                parts = line.split()
                if len(parts) > 2:
                    try:
                        ip = parts[2].strip("()")
                        ipaddress.ip_address(ip)
                        hops_for_this_ip.append(ip)
                    except ValueError:
                        pass

            if hops_for_this_ip and hops_for_this_ip[-1] == target_ip:
                return hops_for_this_ip
        return None

    @network_stable_retry(max_retries=5, consistency_checks=2, delay=2.0)
    def traceroute_names(self, machine_1_name: str, machine_2_name: str) -> list[str] | None:
        """
        Get traceroute path using machine names instead of IPs.
        
        Args:
            machine_1_name (str): Source machine name
            machine_2_name (str): Destination machine name
            
        Returns:
            list[str]|None: List of machine names in the path or None if unreachable
        """
        # Get IP-based traceroute
        ip_hops = self.traceroute(machine_1_name, machine_2_name)
        if not ip_hops:
            return None
            
        # Convert IPs to machine names
        machine_hops = []
        for hop_ip in ip_hops:
            # Find machine with this IP
            hop_machine = None
            for machine_name, ips_set in self.get_machines_ips().items():
                for ip_addr, cidr, _ in ips_set:
                    if ip_addr == hop_ip:
                        hop_machine = machine_name
                        break
                if hop_machine:
                    break
            machine_hops.append(hop_machine if hop_machine else "*")
            
        return machine_hops
    
    def get_special_purpose_ips(self) -> list:
        """
        Get all special-purpose IP addresses in the network.
        
        Returns:
            list: Tuples of (network, machine_name) for special IPs
        """
        special_ips = []
        machines_ips = self.get_machines_ips()
        
        for machine_name, ips in machines_ips.items():
            for ip, cidr, is_special in ips:
                if is_special:
                    special_ips.append((f"{ip}/{cidr}", machine_name))
        
        return special_ips
    
    @network_stable_retry(max_retries=3, consistency_checks=2, delay=1.0)
    def can_ping_without_hop(self, m1: str, m2: str) -> bool:
        """
        Check if two machines can communicate directly without intermediate hops.
        
        Args:
            m1 (str): Name of first machine
            m2 (str): Name of second machine
            
        Returns:
            bool: True if direct communication is possible
        """
        trace = self.traceroute(m1, m2)
        # (f"Traceroute from {m1} to {m2}: {trace}")
        if not trace:
            return False
    
        # check if it has no intermediate devices.
        return len(trace) == 1

    # ------------------------------------------------------------------
    # Forwarding-plane validation (used to grade traceroute answers).
    #
    # A traceroute has no single correct answer: with ECMP or multiple
    # destination interfaces several paths are equally valid. Instead of
    # comparing against one sampled path, we accept any path the network
    # would actually forward, i.e. where every hop is a legitimate FIB
    # next-hop of the previous node toward the destination.
    # ------------------------------------------------------------------
    def is_valid_traceroute(self, src: str, dst: str, trace: List[str]) -> bool:
        """
        Return True if `trace` is a valid forwarding path from `src` to `dst`.

        A path is valid when it starts at `src`, ends at `dst`, has no repeated
        node, and is a consistent forwarding path toward at least one of
        `dst`'s IPv4 addresses: for every consecutive pair (a, b), `b` must be
        a legitimate next-hop of `a` per a's kernel FIB.

        Args:
            src (str): Source machine name.
            dst (str): Destination machine name.
            trace (List[str]): Ordered machine names, source first, dest last.

        Returns:
            bool: True if the path is a valid forwarding path.
        """
        if not trace or trace[0] != src or trace[-1] != dst:
            return False
        if len(set(trace)) != len(trace):
            return False  # a forwarding path toward a destination is loop-free

        targets = [
            ip for ip, _cidr, is_special in self.get_machines_ips().get(dst, set())
            if not is_special and self._is_ipv4(ip)
        ]
        # Accept if the path is a valid forwarding path toward any dst IP.
        return any(self._path_valid_toward(trace, ip) for ip in targets)

    def _path_valid_toward(self, trace: List[str], target_ip: str) -> bool:
        """Check every hop of `trace` is a valid next-hop toward `target_ip`."""
        for a, b in zip(trace, trace[1:]):
            if b not in self._nexthop_nodes(a, target_ip):
                return False
        return True

    def _nexthop_nodes(self, node: str, target_ip: str) -> Set[str]:
        """
        Names of the machines that are valid IP next-hops from `node` toward
        `target_ip`, resolved by longest-prefix match on `node`'s FIB.
        """
        target = ipaddress.ip_address(target_ip)

        best_len = -1
        best_gateways: List[Optional[str]] = []
        for network, gateways in self._routing_table(node):
            if target in network and network.prefixlen > best_len:
                best_len = network.prefixlen
                best_gateways = gateways

        owner = self._ip_owner_map()
        nodes: Set[str] = set()
        for gw in best_gateways:
            # On-link route (gw is None): the destination IP is directly
            # reachable, so the next hop is the machine that owns it.
            name = owner.get(str(target)) if gw is None else owner.get(gw)
            if name:
                nodes.add(name)
        return nodes

    def _routing_table(self, node: str) -> List[Tuple[ipaddress.IPv4Network, List[Optional[str]]]]:
        """Parsed IPv4 FIB of `node` (memoized): list of (network, gateways)."""
        if node not in self._routing_tables:
            output = self.execute_command(machine_name=node, command="ip route") or ""
            self._routing_tables[node] = self._parse_ip_route(output)
        return self._routing_tables[node]

    @staticmethod
    def _parse_ip_route(output: str) -> List[Tuple[ipaddress.IPv4Network, List[Optional[str]]]]:
        """
        Parse `ip route` text into (network, gateways) entries.

        `gateways` holds next-hop IP strings, or `None` for an on-link route.
        Multipath (ECMP) routes contribute several gateways via continuation
        lines. Non-forwarding route types (blackhole, unreachable, ...) are
        skipped. `ip route` lists IPv4 routes only, so prefixes are IPv4.
        """
        _SKIP = {"unreachable", "blackhole", "prohibit", "broadcast",
                 "local", "throw", "nat", "multicast"}
        entries: List[Tuple[ipaddress.IPv4Network, List[Optional[str]]]] = []
        current: Optional[List[Optional[str]]] = None

        for raw in output.splitlines():
            if not raw.strip():
                continue
            if raw[0].isspace():
                # Continuation line of a multipath route: "nexthop via G dev D".
                line = raw.strip()
                if current is not None and line.startswith("nexthop"):
                    m = re.search(r"\bvia (\S+)", line)
                    current.append(m.group(1) if m else None)
                continue

            line = raw.strip()
            dest = line.split()[0]
            if dest in _SKIP:
                current = None
                continue
            network = KatharaClient._to_network(dest)
            if network is None:
                current = None
                continue
            gateways: List[Optional[str]] = []
            via = re.search(r"\bvia (\S+)", line)
            if via:
                gateways.append(via.group(1))
            elif " dev " in line:
                gateways.append(None)  # on-link
            # else: multipath header — gateways arrive on continuation lines.
            current = gateways
            entries.append((network, gateways))
        return entries

    @staticmethod
    def _to_network(token: str) -> Optional[ipaddress.IPv4Network]:
        if token == "default":
            return ipaddress.ip_network("0.0.0.0/0")
        try:
            if "/" not in token:
                token += "/32"
            return ipaddress.ip_network(token, strict=False)
        except ValueError:
            return None

    @staticmethod
    def _is_ipv4(ip: str) -> bool:
        try:
            return ipaddress.ip_address(ip).version == 4
        except ValueError:
            return False

    def _ip_owner_map(self) -> Dict[str, str]:
        """Memoized reverse map {ip_address: machine_name}."""
        if self._ip_owner is None:
            owner: Dict[str, str] = {}
            for name, ips in self.get_machines_ips().items():
                for ip, _cidr, _special in ips:
                    owner[ip] = name
            self._ip_owner = owner
        return self._ip_owner