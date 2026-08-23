LEVEL_FLEXIBILITY_THRESHOLD = 1


from structures.structures import Concept
from enum import Enum



class ZPDNode:
	def __init__(self, concept: Concept, level: int) -> None:
		self.concept = concept
		self.prerequisites: set[ZPDNode] = set()
		self.level: int = level


class GraphHasCycleError(Exception):
	pass


class ZPDGraph:
	def __init__(self, concepts: list[tuple[Concept, int, set[Concept]]]) -> None:
		# Validated BEFORE any ZPDNode/graph state is built: since this is a batch
		# constructor (no incremental add-and-rollback like the old edge-by-edge
		# add_prerequisites), checking first means a rejected batch leaves nothing
		# constructed at all, rather than needing to undo partially-built state.
		# Operates directly on raw Concepts -- ZPDNodes don't exist yet at this point.
		if self._batch_has_cycle(concepts):
			raise GraphHasCycleError("the given concepts/prerequisites contain a cycle")

		self.concept_id_to_node: dict[int, ZPDNode] = {}
		self.relevant_level_nodes: list[set[ZPDNode]] \
				= [set() for _ in range(LEVEL_FLEXIBILITY_THRESHOLD + 1)]
		self.current_level: int = 1
		self.eligible_for_next_level: bool = False
		self.unreachable: set[ZPDNode] = set()
		self.eligible: set[ZPDNode] = set()
		self.completed: set[ZPDNode] = set()

		# Pass 1: register every concept as a ZPDNode first, and initialise data structures.
		for concept, level, prereqs in concepts:
			node = self._add_concept(concept, level)
			if not prereqs and level == 1:
				self.eligible.add(node)
				self.relevant_level_nodes[0].add(node)
			else:
				self.unreachable.add(node)

		# Pass 2: now that every node exists, wire up the actual ZPDNode prerequisite edges.
		for concept, _, prereqs in concepts:
			node = self.concept_id_to_node[concept.id]
			for prereq in prereqs:
				node.prerequisites.add(self.concept_id_to_node[prereq.id])


	@staticmethod
	def _batch_has_cycle(concepts: list[tuple[Concept, int, set[Concept]]]) -> bool:
		# Cycle check over raw Concepts (no ZPDNodes yet). Builds a plain
		# concept_id -> set[concept_id] prerequisite map, then does a DFS from EVERY
		# concept (not just roots) -- tracking nodes currently on the active path (not
		# just ever-visited) so a cycle is detected as soon as the traversal revisits
		# something still "in progress" on this path. Starting only from concepts with
		# no prerequisites ("roots") misses a pure cycle with no root at all (e.g.
		# A->B->C->A and nothing else) -- that case has zero roots, so a root-only scan
		# never even starts a traversal and silently reports no cycle. The shared
		# `visited` set (persisting across every starting concept) keeps this efficient:
		# a concept already fully explored from one start is never redundantly
		# re-explored from another.
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

		return any(_has_cycle_from(concept_id, set()) for concept_id in prereq_ids)


	def _add_concept(self, concept: Concept, level: int) -> ZPDNode:
		# The only way to register a new concept in this graph.
		if concept.id in self.concept_id_to_node:
			return self.concept_id_to_node[concept.id]
		node = ZPDNode(concept, level)
		self.concept_id_to_node[concept.id] = node
		self.relevant_level_nodes[level].add(node)
		return node


	def all_nodes(self) -> set:
		return set(self.concept_id_to_node.values())


	def nodes_at_level(self, level: int) -> set[ZPDNode]:
		nodes: set[ZPDNode] = set()
		for node in self.all_nodes():
			if node.level == level:
				nodes.add(node)
		return nodes


	def complete_concept(self, concept: Concept | int):
		concept_id = concept if isinstance(concept, int) else concept.id
		node = self.concept_id_to_node[concept_id]

		self.eligible.remove(node) # throws if the concept wasnt even eligible
		self.completed.add(node)

		if node.level == self.current_level:
			self.eligible_for_next_level = True
		affected_relative_level = self.current_level - node.level
		self.relevant_level_nodes[affected_relative_level].remove(node)
		if self.eligible_for_next_level
			