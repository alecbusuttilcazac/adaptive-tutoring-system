from structures.structures import Concept


class ZPDNode:
	def __init__(self, concept: Concept) -> None:
		self.concept = concept
		self.prerequisites: set[ZPDNode] = set()


class ZPDGraph:
	def __init__(self) -> None:
		self.concept_id_to_node: dict[int, ZPDNode] = {}

	def _has_cycles(self, child: ZPDNode, parents: set[ZPDNode]) -> bool:
		# We need to check if the prerequisites of a parent ever lead us back to the child.
		# `visited` is shared (via closure) across every recursive call in this whole
		# _has_cycles invocation -- without it, a pre-existing cycle in the graph would
		# make _iterate_prerequisites recurse forever (A's prereqs include B, B's include
		# A, so exploring A -> B -> A -> B... never terminates on its own).
		visited: set[ZPDNode] = set()

		def _iterate_prerequisites(node: ZPDNode) -> tuple[set[ZPDNode], bool]:
			if node in visited:
				# Already explored this node in this search -- stop here instead of
				# re-descending into it (that is what would cause the infinite loop).
				return set(), False
			visited.add(node)

			if node == child:
				# Found child -- short-circuit here, no need to keep exploring node's
				# own prerequisites, since we already have our answer for this branch.
				return {node}, True

			reachable = {node}
			for prereq in node.prerequisites:
				sub_reachable, found = _iterate_prerequisites(prereq)
				reachable |= sub_reachable
				if found:
					# Short-circuit across siblings too: no need to check the rest of
					# node.prerequisites once one branch has already found child.
					return reachable, True
			return reachable, False

		for parent in parents:
			_, found_child = _iterate_prerequisites(parent)
			if found_child:
				return True
		return False
			

	def add_prerequisites(self, child: ZPDNode | Concept, parents: set[ZPDNode] | set[Concept]) \
	-> bool:
		if len(parents) == 0:
			return False

		if isinstance(child, Concept):
			child = self.concept_id_to_node[child.id]

		# Reassigning the loop variable `parent` below only rebinds that local name --
		# it never writes back into `parents` itself (Python has no by-reference iteration
		# the way C++'s `for (auto& x : ...)` does). So any Concept -> ZPDNode resolution
		# needs to be collected into a new set explicitly, not read back from `parents`.
		resolved_parents: set[ZPDNode] = set()
		for parent in parents:
			if isinstance(parent, Concept):
				parent = self.concept_id_to_node[parent.id]
			resolved_parents.add(parent)

		# Checked BEFORE any edge is committed: previously, edges were added first and
		# _has_cycles was only checked afterward -- so a "rejected" call still corrupted
		# the graph, since the already-added edges were never undone. Checking first means
		# a rejected call leaves child.prerequisites completely unchanged.
		if self._has_cycles(child, resolved_parents):
			return False

		for parent in resolved_parents:
			child.prerequisites.add(parent)

		return True
			
			
			
		