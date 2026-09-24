"""Child process that measures a benchmark's jobs one at a time.

Started by :func:`benchmarks.benchmark_common.run_jobs` as
``python -m benchmarks._worker path/to/main.py`` with a JSON list of jobs on
stdin. It imports the script -- whose ``BENCHMARK`` names the driver that built
it -- and for each job writes one JSON line to stdout: ``{"started": job}``
when the measurement begins, then ``{"job", "status", "result" | "error"}``
when it ends. Anything else the measurement prints is sent to stderr so that
stdout carries the protocol alone. Whether a job finishes at all is the
parent's decision: it kills this process when the deadline passes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def load_benchmark(script_path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BENCHMARK


def main() -> None:
    script_path = Path(sys.argv[1]).resolve()
    jobs = json.load(sys.stdin)

    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    def emit(message: dict) -> None:
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    bench = load_benchmark(script_path)
    driver = sys.modules[type(bench).__module__]
    prepare = getattr(driver, "prepare", None)
    for job in jobs:
        if prepare is not None:
            prepare(bench, job)
        emit({"started": job})
        try:
            result = driver.measure(bench, job)
        except TimeoutError:
            emit({"job": job, "status": "timeout"})
        except Exception as exc:  # noqa: BLE001  -- reported to the parent
            emit(
                {
                    "job": job,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            emit({"job": job, "status": "ok", "result": result})


if __name__ == "__main__":
    main()
