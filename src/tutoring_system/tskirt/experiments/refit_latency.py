# Measures how long a T-SKIRT theta refit takes as a concept's response history grows, refitting
# at every batch boundary (every BATCH_SIZE responses) with the same call the pipeline makes
# (fit_concept over the whole history). Writes refit_latency.png and refit_latency.csv next to
# this file. Run on an otherwise idle machine; background load inflates the numbers.
# Run: python -m tutoring_system.tskirt.experiments.refit_latency

import time
from pathlib import Path

import numpy as np

from tutoring_system.structures.structures import Area
from tutoring_system.tskirt.experiments.sigma_vs_T import simulate_qas
from tutoring_system.tskirt.tskirt import fit_concept
from utils.config import config
from utils.plotting import *


MAX_RESPONSES = 200  # history length at the last measurement
REPEATS = 9  # timed refits per measurement; the chart shows their median and interquartile range
MARKER_LIMIT = 60  # above this many points, only the final point gets a marker
OUT_DIR = Path(__file__).parent


def measure() -> tuple[np.ndarray, np.ndarray]:
    # Returns (responses at each measurement, refit times in ms with shape (points, REPEATS)).
    batch = config["BATCH_SIZE"]
    qas, _ = simulate_qas(np.linspace(-1.0, 1.5, MAX_RESPONSES), np.random.default_rng(0))
    concept_id = qas[0].question.concept_id

    area = Area(0)
    sizes, times = [], []
    for start in range(0, MAX_RESPONSES, batch):
        area.history.extend(qas[start:start + batch])
        if start == 0:
            fit_concept(area, concept_id)  # untimed warm-up; the first call in a process is slow
        runs = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fit_concept(area, concept_id)
            runs.append((time.perf_counter() - t0) * 1000)
        sizes.append(len(area.history))
        times.append(runs)
    return np.array(sizes), np.array(times)



def plot(sizes: np.ndarray, times: np.ndarray, path: Path) -> None:
    median = np.median(times, axis=1)
    lo, hi = np.percentile(times, [25, 75], axis=1)

    fig, ax = new_chart(
        "T-SKIRT theta refit latency vs. history length",
        f"Median of {REPEATS} refits per point; band shows the 25th to 75th percentile",
        f"Responses so far (one refit per batch of {config['BATCH_SIZE']})",
        "Refit latency (ms)")
    draw_line(ax, sizes, median, band=(lo, hi),
              markers="all" if len(sizes) <= MARKER_LIMIT else "end")
    ax.set_xlim(left=0)
    label_end(ax, sizes, median, f"{median[-1]:.0f} ms\nat {sizes[-1]} responses")
    save_chart(fig, ax, path)


def main() -> None:
    sizes, times = measure()
    median = np.median(times, axis=1)

    np.savetxt(OUT_DIR / "refit_latency.csv",
               np.column_stack([sizes, median, times.min(axis=1), times.max(axis=1)]),
               delimiter=",", header="responses,median_ms,min_ms,max_ms", comments="", fmt="%.3f")
    plot(sizes, times, OUT_DIR / "refit_latency.png")

    slope = np.polyfit(sizes, median, 1)[0]
    print(f"{'responses':>9} {'median ms':>10} {'min ms':>8} {'max ms':>8}")
    for i in sorted({0, len(sizes) // 4, len(sizes) // 2, 3 * len(sizes) // 4, len(sizes) - 1}):
        print(f"{sizes[i]:>9} {median[i]:>10.1f} {times[i].min():>8.1f} {times[i].max():>8.1f}")
    print(f"linear trend: {slope:.3f} ms per extra response")
    print(f"wrote {OUT_DIR / 'refit_latency.png'} and refit_latency.csv")


if __name__ == "__main__":
    main()
