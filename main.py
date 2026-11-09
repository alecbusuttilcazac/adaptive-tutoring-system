from utils import *
from utils.config import config
from tutoring_system.pipeline import build_mock_area, run_session

N_EXERCISES = 10


def main():
	seed = set_seed(config["SEED"])
	_, area = build_mock_area()
	run_session(area, n_exercises=N_EXERCISES)

	print(f"\nSeed {seed}: {len(area.history)} responses over up to {N_EXERCISES} exercises")
	for kc_id, thetas in sorted(area.theta_by_kc.items()):
		print(f"  kc {kc_id}: theta {thetas[-1]:+.2f} after {len(thetas)} responses")
	mastered = sorted(node.kc.id for node in area.zpd_bandit.graph.completed)
	print(f"  mastered kcs: {mastered or 'none'}")


if __name__ == "__main__":
	main()
