import json
import logging
import os
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import List, Tuple, Union, Type, Optional
from langchain.chat_models.base import BaseChatModel
from pydantic import ValidationError
from tqdm import tqdm

from core.kathara_client import KatharaClient
from core import ground_truth_cache
from questions.base_question import BaseQuestion
from solvers.base_solver import BaseSolver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Error taxonomy populated when a solver fails / produces an unusable answer.
ERROR_TYPE_NONE = None  # successful + populated answer
ERROR_TYPE_EMPTY = "empty_response"        # LLM returned blank content
ERROR_TYPE_JSON_PARSE = "json_parse_error"  # Pydantic/structured-output rejection or json.JSONDecodeError
ERROR_TYPE_RECURSION = "tool_loop_exceeded"  # LangGraph hit recursion_limit
ERROR_TYPE_KATHARA = "kathara_error"        # network probe / kathara client failure
ERROR_TYPE_TIMEOUT = "timeout"              # solver.solve() exceeded the per-run timeout
ERROR_TYPE_OTHER = "other"

# Default per-run wall-clock budget. A run that hangs (e.g. provider accepts the
# connection but never responds) is aborted after this many seconds instead of
# blocking the whole experiment forever. Configurable via run_analysis().
DEFAULT_RUN_TIMEOUT_S = 3600  # 1 hour

# Transient provider errors worth one automatic retry before giving up: a single
# Ollama EOF / dropped connection shouldn't lose a run when a retry would succeed.
_TRANSIENT_ERROR_MARKERS = ("eof", "connection", "timed out", "timeout", "503", "502", "reset by peer")


def _is_transient_error(exc: BaseException) -> bool:
    """Heuristic: is this exception a transient provider/network blip worth a retry?"""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)


def _classify_exception(exc: BaseException) -> str:
    """Bucket an exception raised during solver.solve() into ERROR_TYPE_*."""
    msg = str(exc)
    cls_name = exc.__class__.__name__
    if isinstance(exc, FuturesTimeout):
        return ERROR_TYPE_TIMEOUT
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return ERROR_TYPE_JSON_PARSE
    if "recursion" in msg.lower() or "GraphRecursionError" in cls_name:
        return ERROR_TYPE_RECURSION
    if "kathara" in cls_name.lower() or "Kathara" in msg:
        return ERROR_TYPE_KATHARA
    return ERROR_TYPE_OTHER


def _check_ollama_model_available(model) -> Tuple[bool, str]:
    """
    Return (is_available, reason) for a ChatOllama model.

    Non-ChatOllama models always return (True, "non-ollama") — we don't
    interfere with OpenAI/etc. since they have different failure modes.

    For ChatOllama: GET {base_url}/api/tags and look for the requested model
    in the returned `models[].name` list. Tolerates the `:latest` suffix.
    """
    cls_name = model.__class__.__name__
    if cls_name != "ChatOllama":
        return True, "non-ollama"

    base_url = getattr(model, "base_url", None) or "http://localhost:11434"
    model_name = getattr(model, "model", None)
    if not model_name:
        return False, "no model attribute on ChatOllama instance"

    try:
        import requests
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        tags = r.json().get("models", [])
        available = {m.get("name", "") for m in tags}
        # Ollama lists names with explicit tags (e.g. "gemma4:e4b"). Try exact
        # and ":latest"-stripped/added variants for robustness.
        candidates = {model_name, f"{model_name}:latest", model_name.removesuffix(":latest")}
        if available & candidates:
            return True, "found"
        return False, f"not present in {base_url} (have: {sorted(available)[:8]}{'...' if len(available) > 8 else ''})"
    except Exception as e:
        return False, f"could not reach {base_url}: {e}"

class AnalysisEngine:

    @staticmethod
    def _make_json_serializable(obj):
        """
        Recursively convert any type objects to strings for JSON serialization.
        """
        if isinstance(obj, type):
            return f"<class '{obj.__name__}'>"
        elif isinstance(obj, dict):
            return {key: AnalysisEngine._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [AnalysisEngine._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(AnalysisEngine._make_json_serializable(item) for item in obj)
        else:
            return obj

    @staticmethod
    def get_model_identifier(model):
        return (
            getattr(model, "_identifying_params", {}).get("model_name")
            or getattr(model, "_identifying_params", {}).get("model")
            or getattr(model, "model", None)
            # ChatBedrockConverse exposes the model id as `model_id`, not `model`.
            or getattr(model, "model_id", None)
            or "unknown_model"
        )

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert a string into a safe filename component."""
        return re.sub(r'[^\w\-.]', '_', name)[:80]

    @staticmethod
    def _atomic_write_json(path: str, payload) -> None:
        """Write JSON atomically: temp file + fsync + os.replace.

        Guarantees the destination is never left half-written: after a power
        loss you get either the previous intact file or the new intact file,
        never a truncated one. Used for both partial_results.json and the
        per-run run_*.json safety-net files.
        """
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    @classmethod
    def _save_execution_log(cls, log_dir: str, lab_name: str, question_text: str,
                            model_name: str, solver_name: str, run_index: int,
                            execution_log: list, result: Optional[dict] = None):
        """Save a solver's execution log to a structured directory.

        Embeds the full per-run result (accuracy, tokens, diff, ground truth,
        response, timing) alongside the steps, so partial_results.json can be
        rebuilt from these per-run files alone if it is lost or corrupted.
        Each run_*.json is therefore a self-contained record.
        """
        question_short = cls._sanitize_filename(question_text[:60])
        model_solver = cls._sanitize_filename(f"{model_name}__{solver_name}")

        log_path = os.path.join(log_dir, lab_name, question_short, model_solver)
        os.makedirs(log_path, exist_ok=True)

        # Start from the full result dict, then pin the identity fields and the
        # steps so the file is well-formed even if `result` is partial/None.
        payload = dict(result or {})
        payload.update({
            "lab_name": lab_name,
            "question": question_text,
            "model": model_name,
            "solver": solver_name,
            "run_index": run_index,
            "steps": execution_log,
        })

        log_file = os.path.join(log_path, f"run_{run_index}.json")
        cls._atomic_write_json(log_file, payload)

    @staticmethod
    def _build_pair_summary(all_results: list) -> list:
        """Compute per-(model, solver) aggregate statistics across all labs and questions.

        Includes runs that failed too — accuracy excludes them implicitly via
        `correct=False`, but token and error counts cover the full sample so the
        dashboard can show error rates honestly.
        """
        from statistics import mean, stdev
        stats = defaultdict(lambda: {
            "times": [], "correct": [],
            "tokens_in": [], "tokens_out": [], "tokens_total": [],
            "errors": defaultdict(int),
            "n_all": 0,
        })

        for r in all_results:
            key = (r["model"], r["solver"])
            stats[key]["n_all"] += 1
            err = r.get("error_type")
            if err:
                stats[key]["errors"][err] += 1
            if not r.get("successful", False):
                # Skip time/token aggregation when the solver crashed —
                # tokens may or may not have been recorded.
                # But still record an error tally above.
                continue
            stats[key]["times"].append(r["solver_elapsed_time"])
            stats[key]["correct"].append(1 if r.get("correct", False) else 0)
            stats[key]["tokens_in"].append(int(r.get("tokens_input", 0) or 0))
            stats[key]["tokens_out"].append(int(r.get("tokens_output", 0) or 0))
            stats[key]["tokens_total"].append(int(r.get("tokens_total", 0) or 0))

        summary = []
        for (model, solver), data in sorted(stats.items()):
            times = data["times"]
            correct = data["correct"]
            n = len(times)
            if n == 0:
                # No successful runs: still emit a row so the dashboard can
                # show the error breakdown for this pair.
                summary.append({
                    "model": model,
                    "solver": solver,
                    "n": 0,
                    "n_all": data["n_all"],
                    "correct": 0,
                    "accuracy_pct": 0.0,
                    "mean_time_s": 0.0,
                    "std_time_s": 0.0,
                    "std_accuracy_pct": 0.0,
                    "mean_tokens_total": 0,
                    "mean_tokens_input": 0,
                    "mean_tokens_output": 0,
                    "sum_tokens_total": 0,
                    "errors": dict(data["errors"]),
                })
                continue
            summary.append({
                "model": model,
                "solver": solver,
                "n": n,
                "n_all": data["n_all"],
                "correct": sum(correct),
                "accuracy_pct": round(sum(correct) / n * 100, 1),
                "mean_time_s": round(mean(times), 2),
                "std_time_s": round(stdev(times), 2) if n > 1 else 0.0,
                "std_accuracy_pct": round(stdev(correct) * 100, 1) if n > 1 else 0.0,
                "mean_tokens_total": round(mean(data["tokens_total"]), 1) if data["tokens_total"] else 0,
                "mean_tokens_input": round(mean(data["tokens_in"]), 1) if data["tokens_in"] else 0,
                "mean_tokens_output": round(mean(data["tokens_out"]), 1) if data["tokens_out"] else 0,
                "sum_tokens_total": sum(data["tokens_total"]),
                "errors": dict(data["errors"]),
            })
        return summary

    @classmethod
    def _build_grouped_results(cls, all_results: list) -> list:
        """
        Group flat results by (lab, question, model, solver) with per-run details.
        """
        groups = defaultdict(lambda: {"runs": []})

        for r in all_results:
            key = (r["lab_name"], r["question"], r["model"], r["solver"])
            groups[key]["runs"].append({
                "run_index": r["run_index"],
                "correct": r["correct"],
                "successful": r["successful"],
                "solver_elapsed_time": r["solver_elapsed_time"],
                "error_message": r["error_message"],
                "error_type": r.get("error_type"),
                "had_empty_response": r.get("had_empty_response", False),
                "tokens_input": r.get("tokens_input", 0),
                "tokens_output": r.get("tokens_output", 0),
                "tokens_total": r.get("tokens_total", 0),
                "n_llm_calls": r.get("n_llm_calls", 0),
                "llm_response": r["llm_response"],
                "verification_diff": r["verification_diff"],
            })

        grouped = []
        for (lab, question, model, solver), data in groups.items():
            runs = data["runs"]
            n_correct = sum(1 for r in runs if r["correct"])
            n_total = len(runs)
            grouped.append({
                "lab_name": lab,
                "question": question,
                "model": model,
                "solver": solver,
                "total_runs": n_total,
                "correct_runs": n_correct,
                "accuracy": f"{n_correct}/{n_total}",
                "accuracy_pct": round(n_correct / n_total * 100, 1) if n_total > 0 else 0,
                "runs": runs,
            })

        return grouped

    @classmethod
    def _load_historical_results(
        cls,
        paths: List[str],
        active_lab_names: List[str],
        active_questions: List[str],
        active_pairs: set,
        repetitions: int,
    ) -> Tuple[list, list]:
        """
        Load historical run records from previous experiments and filter them
        for compatibility with the current run.

        Compatibility rules:
          - Only keep runs whose `lab_name` is in the current lab set.
          - Only keep runs whose `question` matches the current question set.
          - Skip any (model, solver) pair already being executed live — the
            new run is authoritative for that pair.
          - For each (lab, question, model, solver), take the first
            `repetitions` runs in order; if fewer exist, keep what we have
            and emit a warning.

        Returns (filtered_records, info_entries) where `info_entries` is a
        list of human-readable provenance lines for the final config block.
        """
        active_labs_set = set(active_lab_names)
        active_questions_set = set(active_questions)
        merged: list = []
        info_entries: list = []

        for path in paths:
            if not os.path.isfile(path):
                logger.warning(f"Historical results file not found, skipping: {path}")
                info_entries.append({"path": path, "status": "missing"})
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load historical file {path}: {e}")
                info_entries.append({"path": path, "status": f"load_error: {e}"})
                continue

            records = payload.get("results", [])
            if not records:
                info_entries.append({"path": path, "status": "no_results"})
                continue

            # Bucket by (lab, question, model, solver), preserve run order.
            buckets: dict = defaultdict(list)
            for r in records:
                if r.get("lab_name") not in active_labs_set:
                    continue
                if r.get("question") not in active_questions_set:
                    continue
                pair = (r.get("model"), r.get("solver"))
                if pair in active_pairs:
                    # New run is replacing this pair — drop history for it.
                    continue
                buckets[(r["lab_name"], r["question"], r["model"], r["solver"])].append(r)

            kept_for_file = 0
            for key, runs in buckets.items():
                # Sort by run_index for determinism, then truncate.
                runs_sorted = sorted(runs, key=lambda x: x.get("run_index", 0))
                if len(runs_sorted) < repetitions:
                    logger.warning(
                        f"Historical {key} has only {len(runs_sorted)} runs "
                        f"(need {repetitions}); keeping all of them."
                    )
                trimmed = runs_sorted[:repetitions]
                # Re-index runs so the merged output is consistent.
                for new_idx, rec in enumerate(trimmed, start=1):
                    rec = dict(rec)
                    rec["run_index"] = new_idx
                    rec.setdefault("error_type", None)
                    rec.setdefault("had_empty_response", False)
                    rec.setdefault("tokens_input", 0)
                    rec.setdefault("tokens_output", 0)
                    rec.setdefault("tokens_total", 0)
                    rec.setdefault("n_llm_calls", 0)
                    rec["_historical_source"] = path
                    merged.append(rec)
                    kept_for_file += 1

            info_entries.append({
                "path": path,
                "status": "ok",
                "records_kept": kept_for_file,
                "pairs_kept": len(buckets),
            })
            logger.info(f"Merged {kept_for_file} historical records from {path}")

        return merged, info_entries

    @classmethod
    def _append_error_log(cls, log_dir: str, *, lab_name: str, question: str,
                          model: str, solver: str, run_index: int,
                          error_type: str, error_message: str,
                          traceback_str: Optional[str] = None,
                          llm_response: Optional[dict] = None) -> None:
        """Append a single error record to log_dir/errors.jsonl."""
        path = os.path.join(log_dir, "errors.jsonl")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lab_name": lab_name,
            "question": question,
            "model": model,
            "solver": solver,
            "run_index": run_index,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback_str,
            "llm_response": llm_response,
        }
        try:
            line = json.dumps(record, default=str)
            # Newline BEFORE the record (not after): if a power loss cuts a
            # write mid-line, the damage is confined to that last line and every
            # prior record stays on its own complete line. The reader
            # (_read_error_log) tolerates a trailing broken line.
            nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
            with open(path, "a", encoding="utf-8") as f:
                f.write(("\n" if nonempty else "") + line)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.warning(f"Failed to append to errors.jsonl: {e}")

    @staticmethod
    def _read_error_log(log_dir: str) -> list:
        """Read errors.jsonl tolerantly, skipping any blank or broken lines.

        An abrupt shutdown can truncate the final line; a naive json-per-line
        parse would raise on it. We skip unparseable lines and keep the rest.
        """
        path = os.path.join(log_dir, "errors.jsonl")
        records = []
        if not os.path.isfile(path):
            return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line in {path}")
        return records

    @classmethod
    def _save_partial_results(cls, all_results: list, failed_labs: list, log_dir: str,
                               generated_at, lab_paths: list, questions: list,
                               llm_solver_pairs: list, repetitions: int):
        """Persist current results to disk so progress is not lost on failure.

        Writes atomically (see _atomic_write_json): a power loss leaves either
        the previous intact file or the new intact file — never a half-written
        / truncated one. This is the fix for the truncation corruption that an
        abrupt shutdown used to cause.
        """
        partial_file = os.path.join(log_dir, "partial_results.json")
        try:
            cls._atomic_write_json(partial_file, {
                "generated_at": generated_at.isoformat(),
                "lab_paths": lab_paths,
                "repetitions": repetitions,
                "results": all_results,
                "failed_labs": failed_labs,
                "total_results": len(all_results),
            })
            logger.debug(f"Partial results saved to: {partial_file}")
        except Exception as e:
            logger.error(f"Failed to save partial results: {e}")

    @classmethod
    def run_analysis(
        cls,
        lab_paths: Union[str, List[str]],
        questions: List[BaseQuestion],
        llm_solver_pairs: List[Tuple[BaseChatModel, Type[BaseSolver]]],
        repetitions: int = 1,
        log_dir: str = None,
        wipe_on_start: bool = True,
        historical_results: Optional[List[str]] = None,
        resume_from: Optional[str] = None,
        run_timeout_s: Optional[float] = DEFAULT_RUN_TIMEOUT_S,
        max_retries: int = 1,
    ):
        """
        Run the analysis with the given questions and LLM/solver pairs across multiple labs.

        run_timeout_s: per-run wall-clock budget for solver.solve(). A run that
            exceeds it is aborted with error_type='timeout' instead of hanging
            the whole experiment. Set to None to disable.
        max_retries: extra attempts on a timeout or transient provider error
            (e.g. Ollama EOF). 1 means one retry (two attempts total).

        Args:
            lab_paths: Single lab path string or list of lab paths.
            questions: List of instantiated question objects.
            llm_solver_pairs: List of (LLM provider, solver class) tuples.
            repetitions: Number of times to repeat each model/solver evaluation.
            log_dir: Directory to save per-run execution logs. If None, logs are
                     saved to experiment_logs/<timestamp>/.
            wipe_on_start: If True, perform a global Kathara wipe before starting.
            resume_from: Path to a partial_results.json from an interrupted run.
                Records matching the current config are restored and the
                corresponding (lab, question, model, solver, run_index) runs
                are skipped. Records for inactive labs/questions/pairs are
                dropped so the resume stays consistent with the current config.

        Returns:
            Dict containing results, grouped_results, and summary organized by lab.
        """
        start = time.time()
        generated_at = datetime.now(timezone.utc)

        # Normalize lab_paths to always be a list
        if isinstance(lab_paths, str):
            lab_paths = [lab_paths]

        # Setup log directory. When resuming, reuse the prior run's directory
        # so new results land alongside the restored ones (partial_results.json,
        # errors.jsonl, per-run logs, and summary.json all stay co-located).
        if log_dir is None:
            if resume_from and os.path.isfile(resume_from):
                log_dir = os.path.dirname(resume_from)
            else:
                timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join("experiment_logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Execution logs will be saved to: {log_dir}")
        
        # Pre-flight: for each unique ChatOllama model, verify it is loaded.
        # Pairs whose model is missing get dropped here so we don't waste time
        # running them and don't pollute partial_results.json with FAIL rows.
        skipped_models: list = []  # list of {model, reason, pairs_dropped}
        seen_check = {}
        kept_pairs = []
        for model, solver_class in llm_solver_pairs:
            mid = cls.get_model_identifier(model)
            if mid in seen_check:
                available, reason = seen_check[mid]
            else:
                available, reason = _check_ollama_model_available(model)
                seen_check[mid] = (available, reason)
            if available:
                kept_pairs.append((model, solver_class))
            else:
                logger.warning(
                    f"Skipping {mid} + {solver_class.__name__}: model unavailable ({reason})"
                )

        # Build the skipped-models summary (deduped, with pair count).
        dropped_counts: dict = defaultdict(int)
        for model, solver_class in llm_solver_pairs:
            mid = cls.get_model_identifier(model)
            if not seen_check.get(mid, (True, ""))[0]:
                dropped_counts[mid] += 1
        for mid, count in dropped_counts.items():
            skipped_models.append({
                "model": mid,
                "reason": seen_check[mid][1],
                "pairs_dropped": count,
            })

        if not kept_pairs:
            logger.error("All model/solver pairs were skipped — nothing to run.")
        llm_solver_pairs = kept_pairs

        if wipe_on_start:
            logger.info("Performing initial Kathara environment wipe...")
            try:
                KatharaClient.wipe()
            except Exception as e:
                logger.warning(f"Initial wipe failed (non-critical): {e}")

        all_results = []
        failed_labs = []
        combination_stats = defaultdict(lambda: {"accuracy_values": [], "elapsed_times": []})

        # Resume support: load prior partial_results.json and mark which
        # (lab, question, model, solver, run_index) tuples are already done.
        # Records for inactive labs/questions/pairs are dropped — the resumed
        # run reflects the *current* config, not the prior one.
        done_keys: set = set()
        resume_info: Optional[dict] = None
        if resume_from:
            resume_info = {"path": resume_from}
            if not os.path.isfile(resume_from):
                logger.warning(f"resume_from not found, ignoring: {resume_from}")
                resume_info["status"] = "missing"
            else:
                active_lab_names_set = {lp.rstrip('/').split('/')[-1] for lp in lab_paths}
                active_questions_set = {q.question_text for q in questions}
                active_pairs_set = {
                    (f"{cls.get_model_identifier(m)} ({m._llm_type})", s.__name__)
                    for m, s in llm_solver_pairs
                }

                # Make an untouched, timestamped backup of the source file
                # BEFORE anything else can overwrite it (the resumed run reuses
                # this same log_dir and will rewrite partial_results.json).
                # This backup is never written to again, so it's a safety net
                # against the corruption-then-overwrite scenario.
                try:
                    backup_ts = generated_at.strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(
                        os.path.dirname(resume_from) or ".",
                        f"partial_results.backup.{backup_ts}.json",
                    )
                    if not os.path.exists(backup_path):
                        shutil.copy2(resume_from, backup_path)
                        logger.info(f"Resume: backed up source to {backup_path}")
                    resume_info["backup_path"] = backup_path
                except Exception as e:
                    logger.warning(f"Resume: could not create backup of {resume_from}: {e}")

                try:
                    with open(resume_from, "r", encoding="utf-8") as f:
                        prior = json.load(f)
                except Exception as e:
                    # A corrupt/truncated resume file must NOT silently restart
                    # from scratch — that would overwrite existing data in this
                    # log_dir. Abort loudly so the user can recover the file
                    # (e.g. rebuild from run_*.json) instead of losing it.
                    resume_info["status"] = f"load_error: {e}"
                    raise RuntimeError(
                        f"resume_from is set but could not be parsed: {resume_from} ({e}). "
                        f"Refusing to start from scratch and overwrite existing results. "
                        f"Recover the file (a backup may exist alongside it, or rebuild it "
                        f"from the per-run run_*.json files) before resuming."
                    ) from e

                if prior is not None:
                    restored = 0
                    dropped = 0
                    for rec in prior.get("results", []):
                        if rec.get("lab_name") not in active_lab_names_set:
                            dropped += 1
                            continue
                        if rec.get("question") not in active_questions_set:
                            dropped += 1
                            continue
                        if (rec.get("model"), rec.get("solver")) not in active_pairs_set:
                            dropped += 1
                            continue
                        key = (
                            rec["lab_name"], rec["question"],
                            rec["model"], rec["solver"], rec.get("run_index"),
                        )
                        if key in done_keys:
                            continue
                        done_keys.add(key)
                        all_results.append(rec)
                        restored += 1
                    resume_info.update({
                        "status": "ok",
                        "records_restored": restored,
                        "records_dropped_inactive": dropped,
                    })
                    logger.info(
                        f"Resume: restored {restored} prior runs from {resume_from} "
                        f"(dropped {dropped} for inactive labs/questions/pairs)"
                    )

        # Progress bar across every (lab × applicable_question × pair × run).
        # We don't know per-lab question applicability until we hit each lab,
        # so the total is an upper bound; tqdm.total is updated downward as
        # non-applicable questions are skipped.
        progress_total = len(lab_paths) * len(questions) * len(llm_solver_pairs) * repetitions
        progress = tqdm(total=progress_total, desc="Runs", unit="run", initial=len(done_keys))

        try:
            for lab_path in lab_paths:
                logger.info(f"Processing lab: {lab_path}")
                lab_name = lab_path.rstrip('/').split('/')[-1]  # Extract lab name from path

                logger.info(f"Instantiating KatharaClient from lab: {lab_path}")
                try:
                    kathara_ctx = KatharaClient.from_lab_path(lab_path=lab_path)
                except Exception as e:
                    error_msg = f"Failed to deploy lab '{lab_name}': {e}"
                    logger.error(error_msg)
                    failed_labs.append({"lab_name": lab_name, "lab_path": lab_path, "error": str(e)})
                    cls._save_partial_results(all_results, failed_labs, log_dir, generated_at, lab_paths, questions, llm_solver_pairs, repetitions)
                    continue

                with kathara_ctx as kathara:
                    logger.info(f"Beginning analysis of questions for lab: {lab_name}")

                    # Pre-compute ground truths for all applicable questions once.
                    # Use a per-lab JSON cache to avoid re-running Kathara probes
                    # every experiment when the lab files haven't changed.
                    applicable_questions = []
                    for question in questions:
                        if not question.applies_to_lab(lab_name):
                            logger.info(f"Skipping question '{question.question_text}' for lab '{lab_name}' (not in whitelist)")
                            # This question won't run any (pair × repetition) here — shrink the bar's total.
                            progress.total -= len(llm_solver_pairs) * repetitions
                            progress.refresh()
                            continue
                        question.inject_client(kathara)

                        cached_payload = ground_truth_cache.load(lab_path, question)
                        if cached_payload is not None:
                            try:
                                ground_truth = question.output_model()(**cached_payload)
                                logger.info(f"GT cache HIT  for '{question.cache_key()}' @ {lab_name}")
                            except Exception as e:
                                logger.warning(
                                    f"GT cache entry for '{question.cache_key()}' @ {lab_name} "
                                    f"failed to rehydrate ({e}); recomputing."
                                )
                                cached_payload = None

                        if cached_payload is None:
                            logger.info(f"GT cache MISS for '{question.cache_key()}' @ {lab_name} — computing")
                            ground_truth = question.get_ground_truth()
                            try:
                                ground_truth_cache.save(lab_path, question, ground_truth)
                            except Exception as e:
                                logger.warning(f"Could not persist GT cache: {e}")

                        applicable_questions.append((question, ground_truth))

                    # Iterate model-first so Ollama keeps each model loaded across all questions.
                    for model, solver_class in llm_solver_pairs:
                        model_identifier = cls.get_model_identifier(model)
                        model_name = f"{model_identifier} ({model._llm_type})"
                        solver_name = solver_class.__name__

                        for question, ground_truth in applicable_questions:
                            logger.info(f"Processing question: '{question.question_text}' for lab: {lab_name}")

                            for run_index in range(1, repetitions + 1):
                                if (lab_name, question.question_text, model_name, solver_name, run_index) in done_keys:
                                    logger.info(
                                        f"Resume skip: {lab_name} | {model_name} + {solver_name} | run {run_index}/{repetitions}"
                                    )
                                    continue
                                solver = solver_class(lab_path=lab_path)
                                logger.info(f"Run {run_index}/{repetitions} using {solver_name} with {model_name} for lab: {lab_name}")
                                progress.set_postfix_str(
                                    f"{lab_name} | {model_identifier} + {solver_name} | run {run_index}/{repetitions}",
                                    refresh=False,
                                )
                                start_solver_time = time.time()

                                response_instance = None
                                error_type = ERROR_TYPE_NONE
                                error_traceback = None
                                successful = False
                                error_message = ""
                                # Attempt the run with a wall-clock timeout, and
                                # retry once on a timeout or transient provider
                                # blip (e.g. Ollama EOF). A fresh solver instance
                                # per attempt avoids carrying over partial state.
                                for attempt in range(max_retries + 1):
                                    if attempt > 0:
                                        logger.warning(
                                            f"Retry {attempt}/{max_retries} for {lab_name} | "
                                            f"{model_name} + {solver_name} | run {run_index} "
                                            f"(prev error_type={error_type})"
                                        )
                                        solver = solver_class(lab_path=lab_path)
                                    try:
                                        logger.debug(f"Solving question '{question.question_text}' for lab: {lab_name} (run {run_index}, attempt {attempt+1})")
                                        if run_timeout_s is None:
                                            response_instance = solver.solve(question=question, model=model)
                                        else:
                                            # Explicit executor (not a `with` block): on timeout we
                                            # must shutdown(wait=False), otherwise the context
                                            # manager would block waiting for the stuck thread and
                                            # defeat the timeout. The abandoned thread leaks until
                                            # solve() returns on its own — Python can't kill it —
                                            # but the experiment proceeds.
                                            ex = ThreadPoolExecutor(max_workers=1)
                                            try:
                                                future = ex.submit(solver.solve, question=question, model=model)
                                                response_instance = future.result(timeout=run_timeout_s)
                                            finally:
                                                ex.shutdown(wait=False)
                                        successful = True
                                        error_type = ERROR_TYPE_NONE
                                        error_message = ""
                                        error_traceback = None
                                        break
                                    except FuturesTimeout as e:
                                        # NOTE: the underlying solve() thread may
                                        # still be running in the background; we
                                        # abandon it. Provider-side this resolves
                                        # when the request eventually completes/dies.
                                        error_message = f"Run exceeded timeout of {run_timeout_s}s"
                                        logger.error(f"Timeout solving with {model_name} for lab {lab_name}: {error_message}")
                                        response_instance = None
                                        successful = False
                                        error_type = ERROR_TYPE_TIMEOUT
                                        error_traceback = traceback.format_exc()
                                        if attempt < max_retries:
                                            continue
                                    except Exception as e:
                                        error_message = f"Unexpected error during solving: {str(e)}"
                                        logger.error(f"Error solving question with {model_name} for lab {lab_name}: {error_message}")
                                        response_instance = None
                                        successful = False
                                        error_type = _classify_exception(e)
                                        error_traceback = traceback.format_exc()
                                        if attempt < max_retries and _is_transient_error(e):
                                            continue
                                        break

                                # Pull diagnostics the solver populated during solve().
                                token_stats = dict(getattr(solver, "last_token_stats", {}) or {})
                                had_empty_response = bool(getattr(solver, "last_had_empty_response", False))

                                if response_instance is not None:
                                    gt_dict = ground_truth.model_dump()
                                    resp_dict = response_instance.model_dump()
                                    # Per-question verification (default: structural
                                    # DeepDiff; traceroute validates the forwarding path).
                                    diff = question.verify(ground_truth, response_instance)
                                    # Ensure type objects are converted to strings
                                    diff = cls._make_json_serializable(diff)
                                    error_message = ""
                                    # If the analyst returned empty content but structured_output
                                    # still produced *something*, flag it without overriding success.
                                    if had_empty_response and error_type is ERROR_TYPE_NONE:
                                        error_type = ERROR_TYPE_EMPTY
                                else:
                                    gt_dict = ground_truth.model_dump()
                                    resp_dict = {}
                                    diff = {}
                                    error_message = "Response did not contain valid output."
                                    logger.warning(
                                        f"Invalid response from {model_name} for question '{question.question_text}' "
                                        f"in lab: {lab_name} (run {run_index})"
                                    )
                                    if error_type is ERROR_TYPE_NONE:
                                        # Solver returned None without raising — count as empty.
                                        error_type = ERROR_TYPE_EMPTY

                                end_solver_time = time.time()
                                solver_elapsed_time = end_solver_time - start_solver_time
                                correct = successful and not diff

                                # Consolidated errors log (one line per failure).
                                if error_type is not None:
                                    cls._append_error_log(
                                        log_dir,
                                        lab_name=lab_name,
                                        question=question.question_text,
                                        model=model_name,
                                        solver=solver_name,
                                        run_index=run_index,
                                        error_type=error_type,
                                        error_message=error_message,
                                        traceback_str=error_traceback,
                                        llm_response=resp_dict,
                                    )

                                result = {
                                    'lab_name': lab_name,
                                    'lab_path': lab_path,
                                    'model': model_name,
                                    'solver': solver_name,
                                    'question': question.question_text,
                                    'ground_truth': gt_dict,
                                    'llm_response': resp_dict,
                                    'verification_diff': diff,
                                    'successful': successful,
                                    'error_message': error_message,
                                    'error_type': error_type,
                                    'had_empty_response': had_empty_response,
                                    'tokens_input': token_stats.get("input_tokens", 0),
                                    'tokens_output': token_stats.get("output_tokens", 0),
                                    'tokens_total': token_stats.get("total_tokens", 0),
                                    'n_llm_calls': token_stats.get("n_llm_calls", 0),
                                    'solver_elapsed_time': solver_elapsed_time,
                                    'run_index': run_index,
                                    'correct': correct,
                                }

                                all_results.append(result)

                                # Save execution log for this run
                                try:
                                    cls._save_execution_log(
                                        log_dir, lab_name, question.question_text,
                                        model_name, solver_name, run_index,
                                        solver.execution_log, result=result,
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to save execution log: {e}")

                                stats_key = (lab_name, question.question_text, model_name, solver_name)
                                combination_stats[stats_key]["accuracy_values"].append(1 if correct else 0)
                                combination_stats[stats_key]["elapsed_times"].append(solver_elapsed_time)

                                # Flush per-run so an interrupted run can resume from this exact point.
                                cls._save_partial_results(all_results, failed_labs, log_dir, generated_at, lab_paths, questions, llm_solver_pairs, repetitions)

                                progress.update(1)
        finally:
            progress.close()
            logger.info("Cleaning up Kathara environment (post-analysis wipe)...")
            try:
                KatharaClient.wipe()
            except Exception as e:
                logger.error(f"Terminal cleanup failed: {e}")

        logger.info(f"Analysis complete. Generated {len(all_results)} live results across {len(lab_paths)} labs.")
        if failed_labs:
            logger.warning(f"{len(failed_labs)} lab(s) failed: {[f['lab_name'] for f in failed_labs]}")

        # Merge historical results from previous experiments. This lets us
        # only re-run the new solver and still compare against legacy baselines.
        merge_info: list = []
        n_live_results = len(all_results)
        if historical_results:
            active_lab_names = [lp.rstrip('/').split('/')[-1] for lp in lab_paths]
            active_questions = [q.question_text for q in questions]
            active_pairs = {
                (f"{cls.get_model_identifier(m)} ({m._llm_type})", s.__name__)
                for m, s in llm_solver_pairs
            }
            historical, merge_info = cls._load_historical_results(
                historical_results,
                active_lab_names=active_lab_names,
                active_questions=active_questions,
                active_pairs=active_pairs,
                repetitions=repetitions,
            )
            all_results.extend(historical)
            logger.info(
                f"Merged {len(historical)} historical records "
                f"(live={n_live_results}, total={len(all_results)})"
            )
            # Rewrite partial_results.json with the merged set (live + historical)
            # so the file on disk matches the final returned results and can be
            # used as a drop-in HISTORICAL_RESULTS input for the next run.
            cls._save_partial_results(
                all_results, failed_labs, log_dir, generated_at,
                lab_paths, questions, llm_solver_pairs, repetitions,
            )

        end = time.time()

        # Build grouped results and pair summary
        grouped_results = cls._build_grouped_results(all_results)
        pair_summary = cls._build_pair_summary(all_results)

        # Save summary overview to log_dir
        summary_file = os.path.join(log_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": generated_at.isoformat(),
                "lab_paths": lab_paths,
                "repetitions": repetitions,
                "pair_summary": pair_summary,
                "failed_labs": failed_labs,
                "historical_sources": merge_info,
                "n_live_results": n_live_results,
                "n_historical_results": len(all_results) - n_live_results,
                "skipped_models": skipped_models,
                "resume_info": resume_info,
            }, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")

        # Return comprehensive results with elapsed_time
        return {
            "config": {
                "lab_paths": lab_paths,
                "questions": [question.question_text for question in questions],
                "llm_solver_pairs": [
                    {
                        "model": cls.get_model_identifier(model),
                        "solver": solver_class.__name__
                    }
                    for model, solver_class in llm_solver_pairs
                ],
                "repetitions": repetitions,
                "generated_at": generated_at.isoformat(),
                "log_dir": log_dir,
                "historical_sources": merge_info,
                "n_live_results": n_live_results,
                "n_historical_results": len(all_results) - n_live_results,
                "skipped_models": skipped_models,
                "resume_info": resume_info,
            },
            "results": all_results,
            "grouped_results": grouped_results,
            "pair_summary": pair_summary,
            "failed_labs": failed_labs,
            "total_results": len(all_results),
            "elapsed_time": end - start
        }
