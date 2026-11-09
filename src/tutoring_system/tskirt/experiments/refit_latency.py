# Measures how long a T-SKIRT theta refit takes as a kc's response history grows, refitting after 
# every batch.

import os
import time
from pathlib import Path

# Must be set before numpy loads: multi-threaded BLAS adds timing jitter to these small problems.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

from tutoring_system.structures.entities import Area
from tutoring_system.tskirt.experiments.sigma_vs_T import simulate_qas
from tutoring_system.tskirt.tskirt import fit_kc
from utils.config import config
from utils.plotting import *


MAX_RESPONSES = 500  # history length at the last measurement
REPEATS = 55  # timed refits per measurement; the chart shows their median and interquartile range
MARKER_LIMIT = 1  # above this many points, only the final point gets a marker
OUT_DIR = Path(__file__).parent


def pin_to_quietest_core() -> None:
    # Timings are much steadier on one idle core: the scheduler can't move the process between
    # cores that are busy or running at different clock speeds. Linux only; skipped elsewhere.
    if not (hasattr(os, "sched_setaffinity") and Path("/proc/stat").exists()):
        return

    def snapshot() -> dict[int, tuple[int, int]]:
        cores = {}
        for line in Path("/proc/stat").read_text().splitlines():
            name, *fields = line.split()
            if name.startswith("cpu") and name[3:].isdigit():
                values = list(map(int, fields))
                cores[int(name[3:])] = (values[3] + values[4], sum(values))  # (idle, total)
        return cores

    before = snapshot()
    time.sleep(1.0)
    after = snapshot()
    busy = {core: 1 - (after[core][0] - before[core][0]) / max(1, after[core][1] - before[core][1])
            for core in os.sched_getaffinity(0) if core in before}
    core = min(busy, key=busy.get)
    os.sched_setaffinity(0, {core})
    print(f"  pinned to core {core} (the least busy, {busy[core]:.0%} load)")


def measure() -> tuple[np.ndarray, np.ndarray]:
    # Returns (responses at each measurement, refit times in ms with shape (points, REPEATS)).
    batch = config["BATCH_SIZE"]
    qas, _ = simulate_qas(np.linspace(-1.0, 1.5, MAX_RESPONSES), np.random.default_rng(0))
    kc_id = qas[0].question.kc_id

    area = Area(0)
    sizes, times = [], []
    started = time.perf_counter()
    for start in range(0, MAX_RESPONSES, batch):
        area.history.extend(qas[start:start + batch])
        if start == 0:
            fit_kc(area, kc_id)  # untimed warm-up; the first call in a process is slow
        runs = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fit_kc(area, kc_id)
            runs.append((time.perf_counter() - t0) * 1000)
        sizes.append(len(area.history))
        times.append(runs)

        done, elapsed = len(area.history), time.perf_counter() - started
        print(f"\r  measuring: {done}/{MAX_RESPONSES} responses ({done / MAX_RESPONSES:.0%}), "
              f"{elapsed:.0f} s elapsed   ", end="", flush=True)
    print()
    return np.array(sizes), np.array(times)



def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    # Least-squares line through the medians. Returns (slope, intercept, R^2).
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
    return slope, intercept, r2


def plot(sizes: np.ndarray, times: np.ndarray, path: Path) -> None:
    median = np.median(times, axis=1)
    slope, intercept, r2 = fit_line(sizes, median)
    lo, hi = np.percentile(times, [25, 75], axis=1)

    fig, ax = new_chart(
        "T-SKIRT theta refit latency vs. history length",
        f"Median of {REPEATS} refits per point; band shows the 25th to 75th percentile",
        f"Responses so far (one refit per batch of {config['BATCH_SIZE']})",
        "Refit latency (ms)")
    draw_line(ax, sizes, median, band=(lo, hi),
              markers="all" if len(sizes) <= MARKER_LIMIT else "end", label="median refit latency")
    draw_line(ax, sizes, slope * sizes + intercept, color=SERIES_COLORS[1], markers="none",
              label=f"linear fit to the median: {slope:.3f} ms per response {intercept:+.1f} ms (R² = {r2:.2f})")
    ax.set_xlim(left=0)
    label_end(ax, sizes, median, f"{median[-1]:.0f} ms\nat {sizes[-1]} responses")
    save_chart(fig, ax, path)


def main() -> None:
    pin_to_quietest_core()
    sizes, times = measure()
    median, mean = np.median(times, axis=1), times.mean(axis=1)

    np.savetxt(OUT_DIR / "refit_latency.csv",
               np.column_stack([sizes, median, mean, times.min(axis=1), times.max(axis=1)]),
               delimiter=",", header="responses,median_ms,mean_ms,min_ms,max_ms", comments="", fmt="%.3f")
    plot(sizes, times, OUT_DIR / "refit_latency.png")

    slope, intercept, r2 = fit_line(sizes, median)
    print(f"{'responses':>9} {'median ms':>10} {'mean ms':>9} {'min ms':>8} {'max ms':>8}")
    for i in sorted({0, len(sizes) // 4, len(sizes) // 2, 3 * len(sizes) // 4, len(sizes) - 1}):
        print(f"{sizes[i]:>9} {median[i]:>10.1f} {mean[i]:>9.1f} {times[i].min():>8.1f} {times[i].max():>8.1f}")
    print(f"linear fit to the median: {slope:.3f} ms per extra response {intercept:+.1f} ms (R^2 = {r2:.3f})")
    print(f"wrote {OUT_DIR / 'refit_latency.png'} and refit_latency.csv")


if __name__ == "__main__":
    main()
