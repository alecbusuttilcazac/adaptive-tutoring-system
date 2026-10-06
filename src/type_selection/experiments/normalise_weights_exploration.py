# Exploration script (not a pytest suite -- no assertions) for watching
# normalise_weights() transform a raw residual list into a capped distribution. Shows the
# intermediate softmax step (what the residuals look like before clipping) alongside the
# final result, since normalise_weights() applies softmax internally and doesn't expose it.
# Run directly: python -m type_selection.experiments.normalise_weights_exploration

import scipy as sp

from type_selection.type_selection import normalise_weights
from config import TYPE_MIN_WEIGHT, TYPE_MAX_WEIGHT, NORM_SOFTMAX_TEMPERATURE


# Residual-realistic scenarios: score_individual_response() produces correct - predicted
# values, which realistically cluster within roughly +-0.15 (see the session's earlier
# sample residuals). This is the scale normalise_weights() actually sees in production.
REALISTIC_SCENARIOS = {
    "one very dominant": [0.13, 0.02, -0.05],
    "one slightly dominant": [0.13, 0.08, -0.05],
    "all roughly equal": [0.01, 0.015, 0.008],
    "two roughly equal": [0.14, 0.13, -0.1],
    "one weak outlier (4)": [0.05, 0.04, 0.03, -0.15],
    "variety 1": [0.2, 0, -0.2],
    "variety 2": [0.2, 0.05, -0.2],
    "variety 3": [0.2, -0.25, -0.2],
    "variety 4": [0.13, 0, -0.13],
    "variety 5": [0.13, -0.05, -0.13],
    "variety 6": [0.13, 0.25, -0.13],
    "variety 7": [0.13, -0.13],
    "variety 8": [0.2, -0.13],
    "variety 9": [0.13, -0.17],
}


def print_scenarios(title: str, scenarios: dict) -> None:
    print(f" {title} \n")
    for name, raw in scenarios.items():
        softmaxed = sp.special.softmax([w / NORM_SOFTMAX_TEMPERATURE for w in raw])
        result = normalise_weights(list(raw))
        print(f"{name}")
        print(f"  raw:                  {[round(float(w), 4) for w in raw]}")
        print(f"  softmax (temp={NORM_SOFTMAX_TEMPERATURE}):  {[round(float(w), 4) for w in softmaxed]}  (sum={sum(softmaxed):.6f})")
        print(f"  result:               {[round(float(w), 4) for w in result]}  (sum={sum(result):.6f})")
        print()

  
def main():
    print(f"TYPE_MIN_WEIGHT={TYPE_MIN_WEIGHT}  TYPE_MAX_WEIGHT={TYPE_MAX_WEIGHT}  "
          f"NORM_SOFTMAX_TEMPERATURE={NORM_SOFTMAX_TEMPERATURE}\n")
    print_scenarios("Realistic (residual-scale) inputs", REALISTIC_SCENARIOS)


if __name__ == "__main__":
    main()
