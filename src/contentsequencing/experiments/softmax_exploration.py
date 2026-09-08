# Exploration script (not a pytest suite -- no assertions) for understanding how
# ZPDBandit.select_concept's softmax distribution behaves across reward scenarios,
# UCB confidence bonuses, and temperatures/scales. Run directly:
#   python -m contentsequencing.experiments.softmax_exploration
#
# Useful for tuning SOFTMAX_TEMPERATURE: too high flattens selection toward uniform
# regardless of reward gaps (defeats the point of reward-weighted selection), too low
# collapses toward pure argmax (no exploration of lower-reward eligible nodes).
#
# Also useful for tuning UCB_SCALE: select_concept doesn't feed raw reward into softmax
# on its own -- update_concept combines it with a UCB confidence bonus first
# (reward + UCB_SCALE * ucb_width). Too large a UCB_SCALE lets the exploration bonus
# swamp the actual learning-progress signal (selection becomes "least-tried wins"
# regardless of reward); too small makes the bonus meaningless noise.

import numpy as np
import scipy as sp

from contentsequencing.bandit import SOFTMAX_TEMPERATURE, UCB_SCALE


def ucb_width(times_exercised: int, t: int) -> float:
    if times_exercised == 0:
        return float("inf")
    return np.sqrt(2 * np.log(t) / times_exercised)


def softmax_probs(values: np.ndarray, temperature: float) -> np.ndarray:
    return sp.special.softmax(values / temperature)


# Representative reward scenarios, roughly matching the scale of real
# reward_function outputs observed from simulated theta trajectories (small,
# often within +-0.15) -- see the sigma_vs_T-style manual exploration this was
# based on.
REWARD_SCENARIOS = {
    "one dominant node": np.array([0.13, -0.01, -0.04, 0.05]),
    "two close contenders": np.array([0.08, 0.075, -0.02, 0.0]),
    "all roughly equal": np.array([0.01, 0.015, 0.008, 0.012]),
    "one clearly regressing": np.array([0.05, 0.04, -0.15, 0.03]),
    "wide spread": np.array([0.15, 0.05, -0.05, -0.15]),
}

TEMPERATURES = [0.5, 0.1, SOFTMAX_TEMPERATURE, 0.02, 0.01]

# Combined reward + UCB scenarios: same nodes as above, but each also carries a
# times_exercised count (all currently-eligible nodes' exercise counts sum to t).
# "lightly tried outsider" is the key case this mechanism targets: a node with a
# strong reward but very little evidence behind it (see the earlier "be a bit weary
# of concepts where only 1 exercise has been taken" discussion).
COMBINED_SCENARIOS = {
    "one lightly-tried, rest well-tried": {
        "rewards": np.array([0.13, -0.01, -0.04, 0.05]),
        "times_exercised": np.array([1, 20, 20, 20]),
    },
    "all equally well-tried": {
        "rewards": np.array([0.13, -0.01, -0.04, 0.05]),
        "times_exercised": np.array([20, 20, 20, 20]),
    },
    "all equally lightly-tried": {
        "rewards": np.array([0.13, -0.01, -0.04, 0.05]),
        "times_exercised": np.array([1, 1, 1, 1]),
    },
}

UCB_SCALES = [0.3, 0.2, UCB_SCALE, 0.05, 0.02]


def print_reward_only_section():
    print("=" * 70)
    print("SECTION 1: raw reward -> softmax, across SOFTMAX_TEMPERATURE values")
    print("=" * 70)
    for name, rewards in REWARD_SCENARIOS.items():
        print(f"\n{name}  (rewards={np.round(rewards, 4).tolist()})")
        for temp in TEMPERATURES:
            probs = softmax_probs(rewards, temp)
            formatted = "  ".join(f"{p:.3f}" for p in probs)
            marker = "  <- current SOFTMAX_TEMPERATURE" if temp == SOFTMAX_TEMPERATURE else ""
            print(f"  temp={temp:<6} probs=[{formatted}]{marker}")


def print_combined_section():
    print("\n" + "=" * 70)
    print("SECTION 2: (reward + UCB_SCALE * ucb_width) -> softmax, across UCB_SCALE")
    print("=" * 70)
    for name, scenario in COMBINED_SCENARIOS.items():
        rewards = scenario["rewards"]
        times_exercised = scenario["times_exercised"]
        t = int(np.sum(times_exercised))
        widths = np.array([ucb_width(n, t) for n in times_exercised])

        print(f"\n{name}")
        print(f"  rewards={np.round(rewards, 4).tolist()}")
        print(f"  times_exercised={times_exercised.tolist()}  (t={t})")
        print(f"  ucb_widths={np.round(widths, 4).tolist()}")

        for scale in UCB_SCALES:
            combined = rewards + scale * widths
            probs = softmax_probs(combined, SOFTMAX_TEMPERATURE)
            formatted = "  ".join(f"{p:.3f}" for p in probs)
            marker = "  <- current UCB_SCALE" if scale == UCB_SCALE else ""
            print(f"  UCB_SCALE={scale:<6} combined={np.round(combined, 4).tolist()}  probs=[{formatted}]{marker}")


def print_exercise_gap_section():
    # Same-sized gap in times_exercised (3 fewer exercises), compared at different
    # absolute scales -- isolates the diminishing-returns shape of 1/sqrt(N) itself,
    # independent of UCB_SCALE or reward. The gap should shrink as both counts grow,
    # since each additional exercise matters less once you already have several.
    print("\n" + "=" * 70)
    print("SECTION 3: UCB width gap for a fixed 3-exercise difference, at various scales")
    print("=" * 70)
    pairs = [(15, 12), (10, 7), (5, 2)]
    for n_high, n_low in pairs:
        t = n_high + n_low
        width_high = ucb_width(n_high, t)
        width_low = ucb_width(n_low, t)
        gap = width_low - width_high
        print(
            f"\nN={n_high} vs N={n_low}  (t={t})"
            f"\n  width(N={n_high}) = {width_high:.4f}"
            f"\n  width(N={n_low}) = {width_low:.4f}"
            f"\n  gap (bonus the less-tried node gets over the well-tried one) = {gap:.4f}"
            f"\n  scaled by current UCB_SCALE={UCB_SCALE}: {UCB_SCALE * gap:.4f}"
        )


def main():
    print_reward_only_section()
    print_combined_section()
    print_exercise_gap_section()


if __name__ == "__main__":
    main()
