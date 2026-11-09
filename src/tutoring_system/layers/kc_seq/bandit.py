import numpy as np

from utils.config import config

DECAY = 0.5 ** (1 / config["REWARD_FUNCTION_HALF_LIFE"])
# Stored oldest -> newest (age descending). Sized to REWARD_FUNCTION_NUM_THETAS so it
# lines up with up to that many diffs, which come from a slice of that many + 1 thetas.
WEIGHTS = DECAY ** np.arange(config["REWARD_FUNCTION_NUM_THETAS"])[::-1]
# for v in WEIGHTS:
# 	print(f"{v:.3f}", end="  ")

import scipy as sp

from tutoring_system.structures.entities import KC
from tutoring_system.layers.kc_seq.zpd import *


# Selects/updates nodes of kcs from the student's ZPD
# Generally a mixture of ZPDES, softmax and UCB1
class ZPDBandit:
	def __init__(self, kcs: list[tuple[KC, int, set[KC]]]) -> None:
		self.graph = ZPDGraph(kcs)


	# Does not include UCB bonus
	@staticmethod
	def reward_function(thetas: np.ndarray) -> float:
		thetas = thetas[-(config["REWARD_FUNCTION_NUM_THETAS"] + 1):]
		diffs = np.diff(thetas)
		if len(diffs) == 0:
			raise RuntimeError()
		
		weights = WEIGHTS[-len(diffs):]
		return float(np.sum(weights * diffs) / np.sum(weights))


	def ucb(self, node: ZPDNode, t: int):
		if node.times_exercised == 0 or t == 0:
			raise RuntimeError()
		return np.sqrt(2 * np.log(t) / node.times_exercised)


	def select_kc(self):
		# Sorted by kc id: set iteration order differs between processes, which would make
		# the choices below irreproducible under a fixed seed.
		nodes = sorted(self.graph.eligible, key=lambda node: node.kc.id)
		untried_kcs = [node for node in nodes if node.times_exercised == 0]
		if untried_kcs:
			return untried_kcs[0]
		
		reward_values = np.array([node.latest_reward for node in nodes]) / config["SOFTMAX_TEMPERATURE"]
		
		probs = sp.special.softmax(reward_values)
		idx = np.random.choice(len(nodes), p=probs)
		
		return nodes[idx]


	def update_kc(self, kc: KC | ZPDNode, thetas: np.ndarray): # UCB-style (absolute)
		node = kc if isinstance(kc, ZPDNode) \
					   else self.graph.kc_id_to_node[kc.id]

		node.times_exercised += 1
		node.latest_reward = self.reward_function(thetas) + config["UCB_SCALE"] * self.ucb(node, sum(r.times_exercised for r in self.graph.eligible))

		if thetas[-1] >= config["MASTERY_THETA"] \
				and node.times_exercised >= config["MIN_EXERCISES"]:
			self.graph.complete_kc(node.kc)