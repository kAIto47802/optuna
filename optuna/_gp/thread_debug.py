from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import atexit
import os
import sys
import time
from typing import Generator


_enabled = os.environ.get("OPTUNA_GP_THREAD_DEBUG") == "1"
_times: defaultdict[str, float] = defaultdict(float)
_counts: defaultdict[str, int] = defaultdict(int)


@contextmanager
def measure(name: str) -> Generator[None, None, None]:
    if not _enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        _times[name] += time.perf_counter() - start
        _counts[name] += 1


def _dump() -> None:
    if not _enabled or len(_times) == 0:
        return

    print("[optuna gp thread debug]", file=sys.stderr)
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        print(f"env.{name}={os.environ.get(name)}", file=sys.stderr)
    try:
        import torch

        print(f"torch.get_num_threads()={torch.get_num_threads()}", file=sys.stderr)
        print(
            f"torch.get_num_interop_threads()={torch.get_num_interop_threads()}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"torch thread info unavailable: {e}", file=sys.stderr)

    for name, total in sorted(_times.items(), key=lambda item: item[1], reverse=True):
        count = _counts[name]
        print(
            f"{name}\tcount={count}\ttotal={total:.6f}\tmean={total / count:.6f}",
            file=sys.stderr,
        )


atexit.register(_dump)
