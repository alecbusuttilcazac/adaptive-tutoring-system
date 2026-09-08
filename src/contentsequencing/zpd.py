LEVEL_FLEXIBILITY_THRESHOLD = 1


from structures.structures import Concept
from enum import Enum


class GraphHasCycleError(Exception):
    pass


class DanglingPrerequisiteError(Exception):
    pass


class LevelInconsistencyError(Exception):
    pass


class ZPDNode:
	def __init__(self, concept: Concept, level: int) -> None:
		self.concept: Concept = concept
		self.prerequisites: set[ZPDNode] = set()
		self.level: int = level

		self.latest_reward: float = 0.0
		self.times_exercised: int = 0


class ZPDGraph:
	def __init__(self, concepts: list[tuple[Concept, int, set[Concept]]]) -> None:
		def _raise_if_dangling_prerequisites(concepts: list[tuple[Concept, int, set[Concept]]]) \
					-> None:
			# Every Concept used as a prerequisite anywhere in the batch must also appear
			# as a top-level entry in that same batch.
			known_ids = {concept.id for concept, _, _ in concepts}
			for concept, _, prereqs in concepts:
				for prereq in prereqs:
					if prereq.id not in known_ids:
						raise DanglingPrerequisiteError(
							f"concept {concept.id} ({concept.name!r}) references prerequisite "
							f"{prereq.id} ({prereq.name!r}), which is not included in this batch"
						)
	
		def _raise_if_level_inconsistent(concepts: list[tuple[Concept, int, set[Concept]]]) -> None:
			# A node's level must be >= every one of its prerequisites' levels. Eg. a level-1
			# node requiring a level-3 prerequisite could never have that prerequisite
			# completed before current_level ever reaches 3.
			levels_by_id = {concept.id: level for concept, level, _ in concepts}
			for concept, level, prereqs in concepts:
				for prereq in prereqs:
					prereq_level = levels_by_id[prereq.id]
					if prereq_level > level:
						raise LevelInconsistencyError(
							f"concept {concept.id} ({concept.name!r}) is at level {level}, but its "
							f"prerequisite {prereq.id} ({prereq.name!r}) is at a higher level "
							f"{prereq_level} -- a node's level must be >= every prerequisite's level"
						)
	
		def _raise_if_has_cycle(concepts: list[tuple[Concept, int, set[Concept]]]) -> None:
			# Cycle check over raw Concepts (no ZPDNodes yet). Builds a plain concept_id -> set
			# [concept_id] prerequisite map, then does a DFS from EVERY concept, tracking nodes 
			# currently on the active path.
			prereq_ids: dict[int, set[int]] = {
				concept.id: {p.id for p in prereqs} for concept, _, prereqs in concepts
			}
			visited: set[int] = set()
	
			def _has_cycle_from(concept_id: int, on_path: set[int]) -> bool:
				if concept_id in on_path:
					return True
				if concept_id in visited:
					return False
				visited.add(concept_id)
				
				on_path = on_path | {concept_id}
				return any(
					_has_cycle_from(prereq_id, on_path)
					for prereq_id in prereq_ids.get(concept_id, set())
				)
	
			if any(_has_cycle_from(concept_id, set()) for concept_id in prereq_ids):
				raise GraphHasCycleError("the given concepts/prerequisites contain a cycle")
	
		
		_raise_if_dangling_prerequisites(concepts)
		_raise_if_level_inconsistent(concepts)
		_raise_if_has_cycle(concepts)

		self.concept_id_to_node: dict[int, ZPDNode] = {}
		self.nodes_by_level: dict[int, set[ZPDNode]] = {}
		self.current_level: int = 1
		self.eligible_for_level: int = 1
		self.unreachable: set[ZPDNode] = set()
		self.eligible: set[ZPDNode] = set()
		self.completed: set[ZPDNode] = set()
		
		def _add_concept(concept: Concept, level: int) -> ZPDNode:
			if concept.id in self.concept_id_to_node:
				return self.concept_id_to_node[concept.id]
			node = ZPDNode(concept, level)
			self.concept_id_to_node[concept.id] = node
			self.nodes_by_level.setdefault(level, set()).add(node)
			return node

		# Pass 1: register every concept as a ZPDNode first, and initialise data structures.
		for concept, level, prereqs in concepts:
			node = _add_concept(concept, level)
			if not prereqs and level == 1:
				self.eligible.add(node)
			else:
				self.unreachable.add(node)

		# Pass 2: now that every node exists, wire up the actual ZPDNode prerequisite edges.
		for concept, _, prereqs in concepts:
			node = self.concept_id_to_node[concept.id]
			for prereq in prereqs:
				node.prerequisites.add(self.concept_id_to_node[prereq.id])


	def all_nodes(self) -> set:
		return set(self.concept_id_to_node.values())


	def nodes_at_level(self, level: int) -> set[ZPDNode]:
		return self.nodes_by_level.get(level, set())


	def complete_concept(self, concept: Concept | ZPDNode) -> None:
		def _advance_level():
			if self._pending_nodes_at_level(self.current_level - LEVEL_FLEXIBILITY_THRESHOLD):
				raise RuntimeError()
			self.current_level += 1
			
			for node in self.nodes_at_level(self.current_level):
				if node.prerequisites <= self.completed: # Subset check
					self.eligible.add(node)
					self.unreachable.remove(node)

		# convert to the respective node
		node = concept if isinstance(concept, ZPDNode) \
					   else self.concept_id_to_node[concept.id]

		self.eligible.remove(node) # throws if the concept wasnt even eligible
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