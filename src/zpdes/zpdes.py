REWARD_FUNCTION_NUM_THETAS = 15 # num thetas to be considered in the reward function.
REWARD_FUNCTION_HALF_LIFE = 5 # half life < k
SOFTMAX_TEMPERATURE = 0.065
CONFIDENCE_PENALTY_THRESHOLD = 4 # < 4 exercises needed to impose penalties
CONFIDENCE_PENALTY_SCALE = 0.025 # absolute per-missing-exercise reward penalty.
MASTERY_THETA_THRESHOLD = 1.25 # minimum theta needed to mark a concept as mastered (completed)

import numpy as np

DECAY = 0.5 ** (1 / REWARD_FUNCTION_HALF_LIFE)
# Stored oldest -> newest (age descending). Sized to REWARD_FUNCTION_NUM_THETAS so it
# lines up with up to that many diffs, which come from a slice of that many + 1 thetas.
WEIGHTS = DECAY ** np.arange(REWARD_FUNCTION_NUM_THETAS)[::-1]
# for v in WEIGHTS:
# 	print(f"{v:.3f}", end="  ")

import scipy as sp

from structures.structures import Concept
from zpdes.zpd import *


class ZPDES:
	def __init__(self, concepts: list[tuple[Concept, int, set[Concept]]]) -> None:
		self.graph = ZPDGraph(concepts)

	
	@staticmethod
	def reward_function(thetas: np.ndarray) -> float:
		thetas = thetas[-(REWARD_FUNCTION_NUM_THETAS + 1):]
		diffs = np.diff(thetas)
		if len(diffs) == 0:
			raise RuntimeError()
		
		weights = WEIGHTS[-len(diffs):]
		return float(np.sum(weights * diffs) / np.sum(weights))


	def confidence_penalty(self, node: ZPDNode):
		return max(0, CONFIDENCE_PENALTY_THRESHOLD - node.times_exercised) \
				* CONFIDENCE_PENALTY_SCALE
	

	def select_concept(self):
		nodes = list(self.graph.eligible)
		reward_values = np.array([node.latest_reward for node in nodes])
		
		probs = sp.special.softmax(reward_values / SOFTMAX_TEMPERATURE)
		idx = np.random.choice(len(nodes), p=probs)
		
		return nodes[idx]


	def update_concept(self, concept: Concept | ZPDNode, thetas: np.ndarray): # UCB-style (absolute)
		node = concept if isinstance(concept, ZPDNode) \
					   else self.graph.concept_id_to_node[concept.id]

		node.times_exercised += 1
		node.latest_reward = self.reward_function(thetas) - self.confidence_penalty(node)

		if thetas[-1] >= MASTERY_THETA_THRESHOLD \
				and node.times_exercised >= CONFIDENCE_PENALTY_THRESHOLD:
			self.graph.complete_concept(node.concept)