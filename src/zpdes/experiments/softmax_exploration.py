# Exploration script (not a pytest suite -- no assertions) for understanding how
# ZPDES.select_concept's softmax distribution behaves across reward scenarios and
# temperatures. Run directly: python -m zpdes.tests.softmax_exploration
#
# Useful for tuning SOFTMAX_TEMPERATURE: too high flattens selection toward uniform
# regardless of reward gaps (defeats the point of reward-weighted selection), too low
# collapses toward pure argmax (no exploration of lower-reward eligible nodes).

import numpy as np
import scipy as sp

from zpdes.zpdes import SOFTMAX_TEMPERATURE


def softmax_probs(rewards: np.ndarray, temperature: float) -> np.ndarray:
    return sp.special.softmax(rewards / temperature)


# Representative reward scenarios, roughly matching the scale of real
# reward_function outputs observed from simulated theta trajectories (small,
# often within +-0.15) -- see the sigma_vs_T-style manual exploration this was
# based on.
SCENARIOS = {
    "one dominant node": np.array([0.13, -0.01, -0.04, 0.05]),
    "two close contenders": np.array([0.08, 0.075, -0.02, 0.0]),
    "all roughly equal": np.array([0.01, 0.015, 0.008, 0.012]),
    "one clearly regressing": np.array([0.05, 0.04, -0.15, 0.03]),
    "wide spread": np.array([0.15, 0.05, -0.05, -0.15]),
}

TEMPERATURES = [0.5, 0.1, SOFTMAX_TEMPERATURE, 0.02, 0.01]


def main():
    for name, rewards in SCENARIOS.items():
        print(f"\n{name}  (rewards={np.round(rewards, 4).tolist()})")
        for temp in TEMPERATURES:
            probs = softmax_probs(rewards, temp)
            formatted = "  ".join(f"{p:.3f}" for p in probs)
            print(f"  temp={temp:<6} probs=[{formatted}]")


if __name__ == "__main__":
    main()
