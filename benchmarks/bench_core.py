#lllllll"""Micro-benchmarks for the LOFOP core engine.

Measures the per-operation cost of the hot paths every subsystem will sit on:
registry lookup/build, config construction/loading/access, event emission,
plugin activation, and error/logging overhead. Numbers are wall-clock via
:mod:`timeit` (GC disabled during timing), reported as best and mean over
several repeats so one-off scheduler noise is visible but does not dominate.

Usage::

    python benchmarks/bench_core.py                    # print report to stdout
    python benchmarks/bench_core.py -o report.md       # also write markdown file
    python benchmarks/bench_core.py --repeats 9        # more repeats, less noise
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import timeit
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lofop.core.config import Config  # noqa: E402
from lofop.core.events import EventBus  # noqa: E402
from lofop.core.exceptions import ConfigError  # noqa: E402
from lofop.core.logging import get_logger  # noqa: E402
from lofop.core.plugins import PluginManager  # noqa: E402
from lofop.core.registry import RegistryHub  # noqa: E402

Case = tuple[str, Callable[[], object]]


@dataclass(frozen=True)
class Result:
    """One benchmark measurement.

    Attributes:
        group: Subsystem the case belongs to.
        name: Human-readable case name.
        number: Inner-loop iterations per repeat.
        best_us: Fastest per-call time across repeats, in microseconds.
        mean_us: Mean per-call time across repeats, in microseconds.
    """

    group: str
    name: str
    number: int
    best_us: float
    mean_us: float

    @property
    def ops_per_sec(self) -> float:
        """Throughput implied by the best per-call time."""
        return 1e6 / self.best_us


def measure(group: str, name: str, fn: Callable[[], object], *, repeats: int) -> Result:
    """Time ``fn`` with timeit autoranging and return per-call statistics."""
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    times = timer.repeat(repeat=repeats, number=number)
    per_call = [t / number * 1e6 for t in times]
    return Result(group, name, number, min(per_call), statistics.mean(per_call))


class _Block:
    """Stand-in component with the constructor shape of a real model block."""

    def __init__(self, size: int = 1, child: object = None, children: Iterable = ()) -> None:
        self.size = size
        self.child = child
        self.children = list(children)


def registry_cases() -> Iterator[Case]:
    hub = RegistryHub()
    models = hub.new("model")
    hub.new("loss").register(_Block, name="cost")
    models.register(_Block, name="block")
    flat = {"type": "block", "size": 3}
    nested = {
        "type": "block",
        "size": 3,
        "child": {"type": "loss/cost", "size": 5},
        "children": [{"type": "loss/cost"}, {"type": "loss/cost"}],
    }
    yield "lookup (get)", lambda: models.get("block")
    yield "build flat spec (2 kwargs)", lambda: models.build(flat)
    yield "build nested spec (3 sub-builds)", lambda: models.build(nested)
    yield "re-register (override)", lambda: models.register(_Block, name="block", override=True)


def config_cases(tmp: Path) -> Iterator[Case]:
    (tmp / "base.yaml").write_text(
        "model:\n  backbone: {depth: 50, width: 1.0}\n  head: {classes: 80}\n"
        "optimizer: {lr: 0.01, momentum: 0.9, weight_decay: 0.0001}\n"
        "data: {root: /data, workers: 8, batch: 16}\n",
        encoding="utf-8",
    )
    (tmp / "exp.yaml").write_text(
        "extends: [base.yaml]\nname: bench\nworkdir: runs/${name}\n"
        "model:\n  backbone: {depth: 101}\n",
        encoding="utf-8",
    )
    exp = tmp / "exp.yaml"
    data = {
        "model": {"backbone": {"depth": 50, "width": 1.0}, "head": {"classes": 80}},
        "optimizer": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
        "data": {"root": "/data", "workers": 8, "batch": 16},
        "tags": ["bench", "core"],
    }
    override = {"model": {"backbone": {"depth": 101}}, "epochs": 12}
    cfg = Config(data)
    yield "construct from dict (11 leaves)", lambda: Config(data)
    yield "load file (extends + interpolation)", lambda: Config.load(exp)
    yield "select dotted path (depth 3)", lambda: cfg.select("model.backbone.depth")
    yield "construct + deep merge", lambda: Config(data).merge(override)
    yield "to_dict", lambda: cfg.to_dict()


def event_cases() -> Iterator[Case]:
    def make_bus(handlers: int) -> EventBus:
        bus = EventBus()
        for _ in range(handlers):
            bus.subscribe("epoch", lambda event: None)
        return bus

    bus0, bus1, bus10 = make_bus(0), make_bus(1), make_bus(10)
    scratch = EventBus()

    def subscribe_unsubscribe() -> None:
        sub = scratch.subscribe("t", lambda event: None)
        scratch.unsubscribe(sub)

    yield "emit, 0 handlers", lambda: bus0.emit("epoch", epoch=1, loss=0.5)
    yield "emit, 1 handler", lambda: bus1.emit("epoch", epoch=1, loss=0.5)
    yield "emit, 10 handlers", lambda: bus10.emit("epoch", epoch=1, loss=0.5)
    yield "subscribe + unsubscribe", subscribe_unsubscribe


def plugin_cases() -> Iterator[Case]:
    def full_cycle() -> None:
        manager = PluginManager(RegistryHub(), EventBus())
        manager.add("p", lambda context: None)
        manager.activate("p")

    yield "manager + add + activate (no-op plugin)", full_cycle


def misc_cases() -> Iterator[Case]:
    log = get_logger("bench")

    def raise_and_catch() -> None:
        try:
            raise ConfigError("missing key", context={"path": "model.depth"})
        except ConfigError:
            pass

    yield "raise + catch ConfigError (with context)", raise_and_catch
    yield "disabled debug log call", lambda: log.debug("epoch %d", 1)


def collect_results(repeats: int) -> list[Result]:
    """Run every benchmark group and return results in execution order."""
    results: list[Result] = []
    with tempfile.TemporaryDirectory() as tmp:
        groups: list[tuple[str, Iterable[Case]]] = [
            ("Registry", registry_cases()),
            ("Config", config_cases(Path(tmp))),
            ("Events", event_cases()),
            ("Plugins", plugin_cases()),
            ("Errors & logging", misc_cases()),
        ]
        for group, cases in groups:
            for name, fn in cases:
                result = measure(group, name, fn, repeats=repeats)
                results.append(result)
                print(
                    f"  {group}: {name}: best {result.best_us:.2f} us/op "
                    f"({result.ops_per_sec:,.0f} ops/s)",
                    file=sys.stderr,
                )
    return results


def environment_lines() -> list[str]:
    """Describe the machine and code revision the numbers came from."""
    cpu = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return [
        f"- **Date:** {now}",
        f"- **Commit:** `{commit}`",
        f"- **Python:** {platform.python_version()}",
        f"- **Platform:** {platform.platform()}",
        f"- **CPU:** {cpu} ({os.cpu_count()} logical cores)",
    ]


def render_report(results: list[Result], repeats: int) -> str:
    """Render the full markdown report."""
    lines = [
        "# LOFOP Core Engine Benchmark Report",
        "",
        "Micro-benchmarks of the task-agnostic core: per-operation wall-clock cost of the",
        "paths that data, model, training, and inference subsystems will run on. Generated",
        "by `benchmarks/bench_core.py`.",
        "",
        "## Environment",
        "",
        *environment_lines(),
        f"- **Method:** timeit autorange, best/mean of {repeats} repeats, GC disabled",
        "",
        "## Results",
        "",
        "| Subsystem | Benchmark | Best (us/op) | Mean (us/op) | Throughput (ops/s) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.group} | {r.name} | {r.best_us:.2f} | {r.mean_us:.2f} "
            f"| {r.ops_per_sec:,.0f} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **Best** is the reproducible cost of the operation; **mean** includes scheduler",
        "  and allocator noise. Compare *best* across commits, watch *mean - best* as a",
        "  noise indicator for the run itself.",
        "- Registry and config costs are **startup/assembly** costs: they are paid once per",
        "  component or experiment, so microseconds here are negligible at run time.",
        "- Event emission is the only core path that can sit inside a training step loop,",
        "  so `emit` should stay in single-digit microseconds for small handler counts.",
        "- `disabled debug log call` bounds the cost of leaving debug logging statements in",
        "  hot paths.",
        "",
        "Regenerate with `python benchmarks/bench_core.py -o <file>` and compare against a",
        "previous report from the same machine; cross-machine comparisons are not meaningful.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LOFOP core engine micro-benchmarks.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="also write the markdown report to this path")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timing repeats per case (default: 5)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    print("Running LOFOP core benchmarks...", file=sys.stderr)
    report = render_report(collect_results(args.repeats), args.repeats)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
