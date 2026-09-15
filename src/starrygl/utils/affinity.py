from __future__ import annotations

import os


def maybe_bind_local_rank_cpu_affinity(*, local_rank: int | None = None, local_world_size: int | None = None) -> dict[str, object]:
    """Bind this process to a per-local-rank CPU slice when requested.

    The launcher sets ``STARRYGL_CPU_AFFINITY=1`` for multi-rank GPU runs.  We
    bind inside the child process because torchrun only sets ``LOCAL_RANK`` for
    the spawned Python workers, not for the parent shell process.
    """

    mode = str(os.environ.get("STARRYGL_CPU_AFFINITY", "0")).strip().lower()
    if mode in {"", "0", "false", "no", "off"}:
        return {"enabled": False}
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {"enabled": False, "reason": "sched_affinity_unavailable"}

    rank = int(os.environ.get("LOCAL_RANK", "0") if local_rank is None else local_rank)
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1") if local_world_size is None else local_world_size)
    world = max(1, world)

    before = sorted(int(v) for v in os.sched_getaffinity(0))
    if not before:
        return {"enabled": False, "reason": "empty_affinity"}

    explicit = str(os.environ.get("STARRYGL_CPU_AFFINITY_MASKS", "")).strip()
    if explicit:
        groups = [part.strip() for part in explicit.split(";")]
        if rank < len(groups) and groups[rank]:
            chosen = _parse_cpu_list(groups[rank])
        else:
            chosen = []
    else:
        offset = int(os.environ.get("STARRYGL_CPU_AFFINITY_OFFSET", "0") or 0)
        cores_per_rank_env = str(os.environ.get("STARRYGL_CPU_AFFINITY_CORES_PER_RANK", "auto")).strip().lower()
        available = before[offset:] if 0 <= offset < len(before) else before
        if cores_per_rank_env in {"", "auto"}:
            cores_per_rank = max(1, len(available) // world)
        else:
            cores_per_rank = max(1, int(cores_per_rank_env))
        start = min(len(available), rank * cores_per_rank)
        stop = min(len(available), start + cores_per_rank)
        chosen = available[start:stop]
        if not chosen:
            chosen = available[rank % len(available) :: world] or available

    allowed = set(before)
    selected = sorted(int(v) for v in chosen if int(v) in allowed)
    if not selected:
        return {"enabled": False, "reason": "no_valid_cpus", "before": before}
    os.sched_setaffinity(0, set(selected))
    return {
        "enabled": True,
        "local_rank": rank,
        "local_world_size": world,
        "before": before,
        "after": selected,
    }


def _parse_cpu_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            begin, end = token.split("-", 1)
            out.extend(range(int(begin), int(end) + 1))
        else:
            out.append(int(token))
    return out


__all__ = ["maybe_bind_local_rank_cpu_affinity"]
