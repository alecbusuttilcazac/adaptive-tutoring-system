from tutoring_system.structures.entities import KC
from enum import Enum
from utils.config import config


class GraphHasCycleError(Exception):
    pass


class DanglingPrerequisiteError(Exception):
    pass


class LevelInconsistencyError(Exception):
    pass


class ZPDNode:
	def __init__(self, kc: KC, level: int) -> None:
		self.kc: KC = kc
		self.prerequisites: set[ZPDNode] = set()
		self.level: int = level

		self.latest_reward: float = 0.0
		self.times_exercised: int = 0


class ZPDGraph:
	def __init__(self, kcs: list[tuple[KC, int, set[KC]]]) -> None:
		def _raise_if_dangling_prerequisites(kcs: list[tuple[KC, int, set[KC]]]) \
					-> None:
			# Every KC used as a prerequisite anywhere in the batch must also appear
			# as a top-level entry in that same batch.
			known_ids = {kc.id for kc, _, _ in kcs}
			for kc, _, prereqs in kcs:
				for prereq in prereqs:
					if prereq.id not in known_ids:
						raise DanglingPrerequisiteError(
							f"kc {kc.id} ({kc.name!r}) references prerequisite "
							f"{prereq.id} ({prereq.name!r}), which is not included in this batch"
						)
	
		def _raise_if_level_inconsistent(kcs: list[tuple[KC, int, set[KC]]]) -> None:
			# A node's level must be >= every one of its prerequisites' levels. Eg. a level-1
			# node requiring a level-3 prerequisite could never have that prerequisite
			# completed before current_level ever reaches 3.
			levels_by_id = {kc.id: level for kc, level, _ in kcs}
			for kc, level, prereqs in kcs:
				for prereq in prereqs:
					prereq_level = levels_by_id[prereq.id]
					if prereq_level > level:
						raise LevelInconsistencyError(
							f"kc {kc.id} ({kc.name!r}) is at level {level}, but its "
							f"prerequisite {prereq.id} ({prereq.name!r}) is at a higher level "
							f"{prereq_level} -- a node's level must be >= every prerequisite's level"
						)
	
		def _raise_if_has_cycle(kcs: list[tuple[KC, int, set[KC]]]) -> None:
			# Cycle check over raw KCs (no ZPDNodes yet). Builds a plain kc_id -> set
			# [kc_id] prerequisite map, then does a DFS from EVERY kc, tracking nodes 
			# currently on the active path.
			prereq_ids: dict[int, set[int]] = {
				kc.id: {p.id for p in prereqs} for kc, _, prereqs in kcs
			}
			visited: set[int] = set()
	
			def _has_cycle_from(kc_id: int, on_path: set[int]) -> bool:
				if kc_id in on_path:
					return True
				if kc_id in visited:
					return False
				visited.add(kc_id)
				
				on_path = on_path | {kc_id}
				return any(
					_has_cycle_from(prereq_id, on_path)
					for prereq_id in prereq_ids.get(kc_id, set())
				)
	
			if any(_has_cycle_from(kc_id, set()) for kc_id in prereq_ids):
				raise GraphHasCycleError("the given kcs/prerequisites contain a cycle")
	
		
		_raise_if_dangling_prerequisites(kcs)
		_raise_if_level_inconsistent(kcs)
		_raise_if_has_cycle(kcs)

		self.kc_id_to_node: dict[int, ZPDNode] = {}
		self.nodes_by_level: dict[int, set[ZPDNode]] = {}
		self.current_level: int = 1
		self.eligible_for_level: int = 1
		self.unreachable: set[ZPDNode] = set()
		self.eligible: set[ZPDNode] = set()
		self.completed: set[ZPDNode] = set()
		
		def _add_kc(kc: KC, level: int) -> ZPDNode:
			if kc.id in self.kc_id_to_node:
				return self.kc_id_to_node[kc.id]
			node = ZPDNode(kc, level)
			self.kc_id_to_node[kc.id] = node
			self.nodes_by_level.setdefault(level, set()).add(node)
			return node

		# Pass 1: register every kc as a ZPDNode first, and initialise data structures.
		for kc, level, prereqs in kcs:
			node = _add_kc(kc, level)
			if not prereqs and level == 1:
				self.eligible.add(node)
			else:
				self.unreachable.add(node)

		# Pass 2: now that every node exists, wire up the actual ZPDNode prerequisite edges.
		for kc, _, prereqs in kcs:
			node = self.kc_id_to_node[kc.id]
			for prereq in prereqs:
				node.prerequisites.add(self.kc_id_to_node[prereq.id])


	def all_nodes(self) -> set:
		return set(self.kc_id_to_node.values())


	def nodes_at_level(self, level: int) -> set[ZPDNode]:
		return self.nodes_by_level.get(level, set())


	def complete_kc(self, kc: KC | ZPDNode) -> None:
		def _advance_level():
			if self._pending_nodes_at_level(self.current_level - config["LEVEL_FLEXIBILITY_THRESHOLD"]):
				raise RuntimeError()
			self.current_level += 1
			
			for node in self.nodes_at_level(self.current_level):
				if node.prerequisites <= self.completed: # Subset check
					self.eligible.add(node)
					self.unreachable.remove(node)

		# convert to the respective node
		node = kc if isinstance(kc, ZPDNode) \
					   else self.kc_id_to_node[kc.id]

		self.eligible.remove(node) # throws if the kc wasnt even eligible
		self.completed.add(node)

		# if the completed node is at the highest current
		# 	level, we are eligible to advance to the next level.
		if node.level >= self.current_level:
			self.eligible_for_level = node.level + 1

		# If eligible to advance, let's check if we can actually do so. Advancing past
		# current_level requires current_level itself to be fully cleared
		while self.eligible_for_level > self.current_level:
			if self._pending_nodes_at_level(self.current_level):
				break # current level has remaining (non-completed) nodes - can not advance past it
			_advance_level()


	def _pending_nodes_at_level(self, level: int) -> set[ZPDNode]:
		return self.nodes_at_level(level) - self.completed