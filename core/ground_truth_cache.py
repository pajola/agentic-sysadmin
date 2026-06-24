from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Legacy basename — no longer written. Kept so the one-off migration script can
# locate old `labs/*/.ground_truth_cache.json` files to move into gt_cache/.
CACHE_FILENAME = ".ground_truth_cache.json"
_HASH_EXTENSIONS = (".conf", ".startup", ".txt")

# core/ground_truth_cache.py -> repo root is two levels up from this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_CACHE_DIR = os.path.join(_REPO_ROOT, "gt_cache")


def is_lab_input_file(rel_path: str) -> bool:
    parts = rel_path.replace(os.sep, "/").split("/")
    if any(p.startswith(".") for p in parts):
        return False
    fname = parts[-1]
    in_subdir = len(parts) > 1
    if not in_subdir and not fname.endswith(_HASH_EXTENSIONS):
        return False
    return True


def _hash_lab(lab_path: str) -> str:
    
    h = hashlib.sha256()
    for root, dirs, files in os.walk(lab_path):
        # Sort for deterministic order; skip hidden dirs (e.g. .git).
        dirs.sort()
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in sorted(files):
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, lab_path).replace(os.sep, "/")
            if not is_lab_input_file(rel):
                continue
            try:
                with open(full, "rb") as f:
                    data = f.read()
            except Exception as e:
                logger.debug(f"GT cache hash skip {rel}: {e}")
                continue
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(data)
            h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _cache_path(lab_path: str) -> str:
    
    lab_basename = os.path.basename(os.path.normpath(lab_path))
    return os.path.join(GT_CACHE_DIR, lab_basename + ".json")


def _load_raw(lab_path: str) -> dict:
    path = _cache_path(lab_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"GT cache unreadable, ignoring: {path} ({e})")
        return {}


def load(lab_path: str, question) -> Optional[Any]:
    """
    Return the cached ground-truth payload (a dict, ready to feed into
    `question.output_model()(**payload)`) or None if absent / invalid.
    """
    cache = _load_raw(lab_path)
    if not cache:
        return None
    lab_hash = _hash_lab(lab_path)
    if cache.get("lab_hash") != lab_hash:
        logger.info(f"GT cache invalidated for {lab_path} (lab hash changed)")
        return None
    key = question.cache_key()
    entry = cache.get("entries", {}).get(key)
    if entry is None:
        return None
    return entry.get("value")


def save(lab_path: str, question, ground_truth) -> None:
    """
    Persist `ground_truth.model_dump()` under `question.cache_key()`.

    If the existing cache has a stale lab_hash, it's reset (we only keep one
    snapshot per lab state — simpler and safer).
    """
    path = _cache_path(lab_path)
    lab_hash = _hash_lab(lab_path)
    cache = _load_raw(lab_path)

    if cache.get("lab_hash") != lab_hash:
        # Fresh hash: discard old entries — they belong to a different snapshot.
        cache = {"lab_hash": lab_hash, "entries": {}}
    cache.setdefault("entries", {})

    key = question.cache_key()
    try:
        value = ground_truth.model_dump()
    except AttributeError:
        # Allow plain dict ground truths for safety.
        value = ground_truth

    cache["entries"][key] = {
        "value": value,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "question_class": question.__class__.__name__,
    }

    try:
        # gt_cache/ may not exist yet (unlike the lab dir, which always did).
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
        os.replace(tmp, path)  # atomic: tmp and path are in the same dir
        logger.debug(f"GT cache wrote entry '{key}' -> {path}")
    except Exception as e:
        logger.warning(f"Failed to write GT cache to {path}: {e}")