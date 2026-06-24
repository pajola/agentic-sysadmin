import json
import logging
import requests
from dotenv import load_dotenv
import os
from langchain.chat_models.base import BaseChatModel, init_chat_model
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse

from core.analysis_engine import AnalysisEngine
# from solvers.agent_solver_from_files import AgentSolverFromFiles
from solvers.bulk_processing_solver_from_files import BulkProcessingSolverFromFiles
from solvers.bulk_reAct_solver_from_files import BulkReactSolverFromFiles
# from solvers.agent_solver_from_files_and_network import AgentSolverFromFilesAndNetwork
from solvers.planner_agent_solver import PlannerAgentSolver, PlannerAgentSolverWithNetwork
from solvers.guided_retrieval_solver import GuidedRetrievalAgentSolver
from questions import *

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Configuration
LAB_PATHS =[
    "labs/lab_medium-nat-web",
    "labs/lab_small-internet-with-dns-and-web-server",
    "labs/lab_alien",
    "labs/lab_stairs",
    "labs/lab_dns-load-balancer-with-rip",
    "labs/lab_medium-ftp-dhcp-web"
]
OUTPUT_FILE = "analysis_results.json"
REPETITIONS = 10

# Optional: paths to partial_results.json files from previous experiments.
# AnalysisEngine will merge runs whose (lab, question) match the current
# config, except for (model, solver) pairs that are being re-run live below
# — those are always taken from this run. This lets us iterate on a single
# solver without re-running all the baselines.
HISTORICAL_RESULTS = [
    # Renamed run with the crashed PlannerAgentSolver records. The planner
    # pair is re-run live below, so the merge drops its (failed) history and
    # keeps only Bulk/React/Strategic from here.
    # "experiment_logs/20260603_164046/partial_results.renamed.json",
]

# Optional: path to a partial_results.json from a previously interrupted run.
# When set, AnalysisEngine restores already-completed (lab, question, model,
# solver, run_index) runs from that file and only executes the missing ones.
# Records for labs/questions/pairs no longer in the current config are dropped.
# RESUME_FROM = "experiment_logs/20260605_130242/partial_results.json"
RESUME_FROM = None

OLLAMA_BASE_URL = "http://localhost:11434"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials not configured, skipping notification.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        logging.warning(f"Failed to send Telegram notification: {e}")


def send_telegram_document(file_path: str, caption: str | None = None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram credentials not configured, skipping document upload.")
        return
    if not os.path.isfile(file_path):
        logging.warning(f"File not found, skipping Telegram upload: {file_path}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(file_path, "rb") as f:
            data = {"chat_id": TELEGRAM_CHAT_ID}
            if caption:
                # Telegram caption max length is 1024 chars
                data["caption"] = caption[:1024]
            requests.post(url, data=data, files={"document": f}, timeout=120)
    except Exception as e:
        logging.warning(f"Failed to send Telegram document: {e}")

def main():
    
    # Configure questions with lab-specific filtering.
    # CanPing and CommonSubnetwork are parametric (need a device pair per lab).
    # We use one pair per lab for both questions; the set is balanced so each
    # question has 3 positive cases (directly connected) and 3 negative cases.
    #   server1/r1  (medium-nat-web)        -> not adjacent  : CanPing=False, CommonSubnet=null
    #   as100r1/as100r3 (small-internet)    -> domain J      : CanPing=True,  CommonSubnet=100.1.0.0/30
    #   as3r1/as3r2 (alien)                 -> domain C      : CanPing=True,  CommonSubnet=3.3.0.0/24
    #   as1r1/as4r1 (stairs)                -> opposite ends : CanPing=False, CommonSubnet=null
    #   web1/web2   (dns-load-balancer-rip) -> domain D      : CanPing=True,  CommonSubnet=30.0.0.0/24
    #   web/dns     (medium-ftp-dhcp-web)   -> DMZ vs servers: CanPing=False, CommonSubnet=null
    questions = [
        CanPingWithoutHopQuestion(m1="server1", m2="r1", lab_whitelist=["lab_medium-nat-web"]),
        CanPingWithoutHopQuestion(m1="as100r1", m2="as100r3", lab_whitelist=["lab_small-internet-with-dns-and-web-server"]),
        CanPingWithoutHopQuestion(m1="as3r1", m2="as3r2", lab_whitelist=["lab_alien"]),
        CanPingWithoutHopQuestion(m1="as1r1", m2="as4r1", lab_whitelist=["lab_stairs"]),
        CanPingWithoutHopQuestion(m1="web1", m2="web2", lab_whitelist=["lab_dns-load-balancer-with-rip"]),
        CanPingWithoutHopQuestion(m1="web", m2="dns", lab_whitelist=["lab_medium-ftp-dhcp-web"]),
        CommonSubnetworkQuestion(m1="as100r1", m2="as100r3", lab_whitelist=["lab_small-internet-with-dns-and-web-server"]),
        CommonSubnetworkQuestion(m1="server1", m2="r1", lab_whitelist=["lab_medium-nat-web"]),
        CommonSubnetworkQuestion(m1="as3r1", m2="as3r2", lab_whitelist=["lab_alien"]),
        CommonSubnetworkQuestion(m1="as1r1", m2="as4r1", lab_whitelist=["lab_stairs"]),
        CommonSubnetworkQuestion(m1="web1", m2="web2", lab_whitelist=["lab_dns-load-balancer-with-rip"]),
        CommonSubnetworkQuestion(m1="web", m2="dns", lab_whitelist=["lab_medium-ftp-dhcp-web"]),
        # Traceroute: one reachable, multi-hop pair per lab (verified live).
        #   client1->server2 (medium-nat-web)   : [client1, r1, r2, server2]
        #   client->webserver (small-internet)  : [client, as200r1, as20r1, as20r2, as100r1, as100r2, webserver]
        #                                          (needs BGP convergence, ~50s — see GT cache note)
        #   pc1->pc2 (alien)                    : [pc1, as4, pc2]   (inter-AS routing is broken by design)
        #   as1r1->as4r1 (stairs)               : [as1r1, as2r1, as3r1, as4r1]
        #   client->nslocal (dns-lb-rip)        : [client, r1, nslocal]  (only nslocal is fully routable)
        #   samba->web (medium-ftp-dhcp-web)    : [samba, r2, web]
        TracerouteQuestion(m1="client1", m2="server2", lab_whitelist=["lab_medium-nat-web"]),
        TracerouteQuestion(m1="client", m2="webserver", lab_whitelist=["lab_small-internet-with-dns-and-web-server"]),
        TracerouteQuestion(m1="pc1", m2="pc2", lab_whitelist=["lab_alien"]),
        TracerouteQuestion(m1="as1r1", m2="as4r1", lab_whitelist=["lab_stairs"]),
        TracerouteQuestion(m1="client", m2="nslocal", lab_whitelist=["lab_dns-load-balancer-with-rip"]),
        TracerouteQuestion(m1="samba", m2="web", lab_whitelist=["lab_medium-ftp-dhcp-web"]),
        CountNodesQuestion(),
        DevicesWithMostIPsQuestion(),
        DevicesWithMultipleIPsQuestion(),
        IPv6AddressesQuestion(),
        SubnetworksQuestion(),
        ZoneTransferQuestion(),
        EnabledServicesQuestion(),
    ]
    
    # Configure LLM models
    qwen35 = ChatOllama(
        model="qwen3.5:9b",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    ministral3= ChatOllama(
        model="ministral-3:14b",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    llama31 = ChatOllama(
        model="llama3.1:latest",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    lfm25_thinking = ChatOllama(
        model="lfm2.5-thinking:latest",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    granite4 = ChatOllama(
        model="granite4:latest",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    rnj1 = ChatOllama(
        model="rnj-1:8b",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    gemma4 = ChatOllama(
        model="gemma4:e4b",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )

    nemotron3_nano = ChatOllama(
        model="nemotron-3-nano:4b",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_ctx=16384,
        num_predict=4000,
    )


    # AWS Bedrock models
    # Credentials are picked up automatically from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    # (and optionally AWS_SESSION_TOKEN) loaded from .env by python-dotenv.
    # Make sure model access is granted in the Bedrock console for the chosen region.
    # NOTE: Claude 4.x and Llama 4 on Bedrock require inference profile IDs
    # (geo-prefixed model IDs like "us.*"), not raw model IDs. The "us." prefix
    # enables cross-region inference within US regions.

    # Kimi K2.5 — Moonshot AI, ON_DEMAND on Bedrock, no inference profile needed.
    bedrock_kimi25 = ChatBedrockConverse(
        model="moonshotai.kimi-k2.5",
        temperature=0.1,
        region_name=AWS_REGION,
    )

    # GLM 5 — Z.AI, ON_DEMAND on Bedrock (verified ACTIVE in us-east-1),
    # no inference profile needed.
    bedrock_glm5 = ChatBedrockConverse(
        model="zai.glm-5",
        temperature=0.1,
        region_name=AWS_REGION,
    )


    # Run the analysis using the new class method approach with multiple labs
    results = AnalysisEngine.run_analysis(
        lab_paths=LAB_PATHS,  # Now supports multiple lab paths
        questions=questions,
        llm_solver_pairs=[  # Now expects solver classes instead of instances
            (qwen35, BulkProcessingSolverFromFiles),
            (qwen35, BulkReactSolverFromFiles),
            (qwen35, PlannerAgentSolver),
            (qwen35, GuidedRetrievalAgentSolver),
            (llama31, BulkProcessingSolverFromFiles),
            (llama31, BulkReactSolverFromFiles),
            (llama31, PlannerAgentSolver),
            (llama31, GuidedRetrievalAgentSolver),
            (ministral3, BulkProcessingSolverFromFiles),
            (ministral3, BulkReactSolverFromFiles),
            (ministral3, PlannerAgentSolver),
            (ministral3, GuidedRetrievalAgentSolver),
            (gemma4, BulkProcessingSolverFromFiles),
            (gemma4, BulkReactSolverFromFiles),
            (gemma4, PlannerAgentSolver),
            (gemma4, GuidedRetrievalAgentSolver),
            (granite4, BulkProcessingSolverFromFiles),
            (granite4, BulkReactSolverFromFiles),
            (granite4, PlannerAgentSolver),
            (granite4, GuidedRetrievalAgentSolver),
            (rnj1, BulkProcessingSolverFromFiles),
            (rnj1, BulkReactSolverFromFiles),
            (rnj1, PlannerAgentSolver),
            (rnj1, GuidedRetrievalAgentSolver),
            (nemotron3_nano, BulkProcessingSolverFromFiles),
            (nemotron3_nano, BulkReactSolverFromFiles),
            (nemotron3_nano, PlannerAgentSolver),
            (nemotron3_nano, GuidedRetrievalAgentSolver),
            (lfm25_thinking, BulkProcessingSolverFromFiles),
            (lfm25_thinking, BulkReactSolverFromFiles),
            (lfm25_thinking, PlannerAgentSolver),
            (lfm25_thinking, GuidedRetrievalAgentSolver)
            (bedrock_kimi25, BulkProcessingSolverFromFiles),
            (bedrock_kimi25, BulkReactSolverFromFiles),
            (bedrock_kimi25, PlannerAgentSolver),
            (bedrock_kimi25, GuidedRetrievalAgentSolver),
            (bedrock_glm5, BulkProcessingSolverFromFiles),
            (bedrock_glm5, BulkReactSolverFromFiles),
            (bedrock_glm5, PlannerAgentSolver),
            (bedrock_glm5, GuidedRetrievalAgentSolver),
        ],
        repetitions=REPETITIONS,
        historical_results=HISTORICAL_RESULTS or None,
        resume_from=RESUME_FROM,
    )

    # When saving the results:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    config_meta = results.get("config", {})
    generated_at = config_meta.get("generated_at", "unknown time")
    print(f"\nAnalysis completed for {len(results['config']['lab_paths'])} labs with {REPETITIONS} repetitions per model/solver pair")
    print(f"Generated at: {generated_at}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Execution logs saved to: {results['config'].get('log_dir', 'N/A')}")

    # Print per-question grouped results
    print("\n" + "="*80)
    print("PER-QUESTION RESULTS (grouped)")
    print("="*80)
    for group in results.get("grouped_results", []):
        status_icons = ["OK" if r["correct"] else "FAIL" for r in group["runs"]]
        print(f"\n  [{group['lab_name']}] {group['question'][:60]}")
        print(f"    {group['model']} + {group['solver']}")
        print(f"    Accuracy: {group['accuracy']} ({group['accuracy_pct']}%)  Runs: {' | '.join(status_icons)}")

    # Print statistics per model+solver pair (computed by AnalysisEngine)
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL + SOLVER PAIR")
    print("="*80)
    for entry in results.get("pair_summary", []):
        print(f"\n{entry['model']} + {entry['solver']}")
        print(f"  Samples: n={entry['n']} (total runs incl. failures: {entry.get('n_all', entry['n'])})")
        print(f"  Time:     mean={entry['mean_time_s']:.2f}s  std={entry['std_time_s']:.2f}s")
        print(f"  Tokens:   mean/run={entry.get('mean_tokens_total', 0):.0f}  total={entry.get('sum_tokens_total', 0):,}")
        print(f"  Accuracy: mean={entry['accuracy_pct']:.1f}%  std={entry['std_accuracy_pct']:.1f}%  ({entry['correct']}/{entry['n']} correct)")
        if entry.get('errors'):
            errs = ", ".join(f"{k}={v}" for k, v in entry['errors'].items())
            print(f"  Errors:   {errs}")

    # Historical merge info
    n_live = results['config'].get('n_live_results', 0)
    n_hist = results['config'].get('n_historical_results', 0)
    if n_hist:
        print("\n" + "="*80)
        print(f"MERGE INFO: {n_live} live + {n_hist} historical = {n_live + n_hist} total records")
        for src in results['config'].get('historical_sources', []):
            print(f"  - {src.get('path')}: {src.get('status')} "
                  f"({src.get('records_kept', 0)} records, {src.get('pairs_kept', 0)} pairs)")
    print("\n" + "="*80)
    print(f"Summary saved to: {results['config']['log_dir']}/summary.json")

    # Report skipped models (preflight check)
    skipped_models = results['config'].get("skipped_models", [])
    if skipped_models:
        print("\n" + "="*80)
        print("SKIPPED MODELS (preflight)")
        print("="*80)
        for sm in skipped_models:
            print(f"  - {sm['model']}: {sm['reason']} (dropped {sm['pairs_dropped']} pair(s))")

    # Report failed labs
    failed_labs = results.get("failed_labs", [])
    log_dir = results.get("config", {}).get("log_dir")
    partial_results_path = os.path.join(log_dir, "partial_results.json") if log_dir else None

    if failed_labs:
        print("\n" + "="*80)
        print("FAILED LABS")
        print("="*80)
        for f in failed_labs:
            print(f"  [{f['lab_name']}] {f['error']}")
        msg = (
            f"net-topology-checker completed with {len(failed_labs)} failed lab(s):\n"
            + "\n".join(f"- {f['lab_name']}: {f['error']}" for f in failed_labs)
            + f"\n\nPartial results saved to: {results['config']['log_dir']}"
        )
        if skipped_models:
            msg += "\n\nSkipped models:\n" + "\n".join(
                f"- {sm['model']} ({sm['pairs_dropped']} pair(s)): {sm['reason']}"
                for sm in skipped_models
            )
        send_telegram_message(msg)
    else:
        msg = (
            f"net-topology-checker completed successfully.\n"
            f"{len(results['config']['lab_paths'])} labs, {REPETITIONS} repetitions.\n"
            f"Results saved to: {OUTPUT_FILE}"
        )
        if skipped_models:
            msg += "\n\nSkipped models:\n" + "\n".join(
                f"- {sm['model']} ({sm['pairs_dropped']} pair(s)): {sm['reason']}"
                for sm in skipped_models
            )
        send_telegram_message(msg)

    if partial_results_path:
        send_telegram_document(
            partial_results_path,
            caption=f"partial_results.json ({os.path.basename(log_dir)})",
        )

if __name__ == "__main__":
    main()