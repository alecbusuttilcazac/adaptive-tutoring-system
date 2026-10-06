import numpy as np

from config import (
	REWARD_FUNCTION_NUM_THETAS,
	REWARD_FUNCTION_HALF_LIFE,
	SOFTMAX_TEMPERATURE,
	UCB_SCALE,
	MASTERY_THETA,
	MIN_EXERCISES,
)

DECAY = 0.5 ** (1 / REWARD_FUNCTION_HALF_LIFE)
# Stored oldest -> newest (age descending). Sized to REWARD_FUNCTION_NUM_THETAS so it
# lines up with up to that many diffs, which come from a slice of that many + 1 thetas.
WEIGHTS = DECAY ** np.arange(REWARD_FUNCTION_NUM_THETAS)[::-1]
# for v in WEIGHTS:
# 	print(f"{v:.3f}", end="  ")

import scipy as sp

from structures.structures import Concept
from contentsequencing.zpd import *


# Selects/updates nodes of concepts from the student's ZPD
# Generally a mixture of ZPDES, softmax and UCB1
class ZPDBandit:
	def __init__(self, concepts: list[tuple[Concept, int, set[Concept]]]) -> None:
		self.graph = ZPDGraph(concepts)


	# Does not include UCB bonus
	@staticmethod
	def reward_function(thetas: np.ndarray) -> float:
		thetas = thetas[-(REWARD_FUNCTION_NUM_THETAS + 1):]
		diffs = np.diff(thetas)
		if len(diffs) == 0:
			raise RuntimeError()
		
		weights = WEIGHTS[-len(diffs):]
		return float(np.sum(weights * diffs) / np.sum(weights))


	def ucb(self, node: ZPDNode, t: int):
		if node.times_exercised == 0 or t == 0:
			raise RuntimeError()
		return np.sqrt(2 * np.log(t) / node.times_exercised)


	def select_concept(self):
		untried_concepts = {node for node in self.graph.eligible if node.times_exercised == 0}
		if untried_concepts:
			return untried_concepts.pop()
		
		nodes = list(self.graph.eligible)
		reward_values = np.array([node.latest_reward for node in nodes]) / SOFTMAX_TEMPERATURE
		
		probs = sp.special.softmax(reward_values)
		idx = np.random.choice(len(nodes), p=probs)
		
		return nodes[idx]


	def update_concept(self, concept: Concept | ZPDNode, thetas: np.ndarray): # UCB-style (absolute)
		node = concept if isinstance(concept, ZPDNode) \
					   else self.graph.concept_id_to_node[concept.id]

		node.times_exercised += 1
		node.latest_reward = self.reward_function(thetas) + UCB_SCALE * self.ucb(node, sum(r.times_exercised for r in self.graph.eligible))

		if thetas[-1] >= MASTERY_THETA \
				and node.times_exercised >= MIN_EXERCISES:
			self.graph.complete_concept(node.concept)