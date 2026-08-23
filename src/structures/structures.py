from enum import Enum

import numpy as np


class Concept:
    def __init__(self, name: str, id: int):
        self.name: str = name
        self.id: int = id

    # Compares/hashes by id, not object identity: without this, two separate Concept
    # instances built from the same id (e.g. loaded fresh from data twice) would be
    # treated as different -- breaking dict-keying (fit_all_concepts) and equality
    # checks (responses_for_concept's qa.question.concept == concept) that rely on
    # "same concept" meaning "same id", not "same Python object in memory".
    def __eq__(self, other):
        return isinstance(other, Concept) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Area: # A student can have different areas of study.
    def __init__(self, id: int):
        # Deferred (function-local) import, not a top-level one: zpdes.py already imports
        # Concept from this module, so a top-level `from zpdes.zpdes import ZPDGraph` here
        # would create a circular import between structures.py and zpdes.py. Importing
        # inside __init__ instead means this only runs once an Area is actually
        # constructed, by which point both modules have finished loading.
        from zpdes.zpdes import ZPDGraph

        self.id: int = id
        # ZPDGraph is now a batch-only constructor (concepts + prerequisites known up
        # front, validated for cycles before any graph state is built) -- an Area starts
        # with an empty graph, populated later once its concepts/prerequisites are known.
        self.graph: ZPDGraph = ZPDGraph([])  # each Area owns its own independent dependency graph
        # "QuestionAnswer" quoted, not imported: importing it here would create a circular
        # import (tskirt.py already imports Concept/Student from this module), and this
        # type hint is documentation only -- nothing in this file actually constructs or
        # calls QuestionAnswer, so there's nothing to import for real.
        all_questions: dict[int, dict[QuestionType, set["Question"]]] = {} # all possible questions
        self.history: set["QuestionAnswer"] = set()
        self.theta_by_concept: dict[int, np.ndarray] = {}  # concept_id -> fitted theta array

    # Compares/hashes by id, same reasoning as Concept.__eq__/__hash__ above -- needed
    # for Area to work correctly as a set element (e.g. Student.areas) or dict key.
    def __eq__(self, other):
        return isinstance(other, Area) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Student:
    def __init__(self, id: int):
        self.id: int = id
        self.areas: set[Area] = set() # A student can have different areas of study.

    # Compares/hashes by id, same reasoning as Concept.__eq__/__hash__ above.
    def __eq__(self, other):
        return isinstance(other, Student) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class QuestionType(Enum):
      MULTIPLE_CHOICE = 0
      TRUE_OR_FALSE = 1
      FILL_IN_THE_BLANK = 2
      FLASHCARD = 3

      def guess_floor(self):
            # Each activity types has a guess_floor parameter (the chance of getting a question
            # right by guessing). In the 3PL-style IRT formula used in `neg_log_likelihood`.
            if self == QuestionType.MULTIPLE_CHOICE:   return 0.25
            elif self == QuestionType.TRUE_OR_FALSE:   return 0.5
            elif self == QuestionType.FILL_IN_THE_BLANK:  return 0.0
            elif self == QuestionType.FLASHCARD:    return 0.0
            else: raise ValueError("Unknown question type: {}".format(self))


class Question:
     def __init__(self, concept: Concept, question_type: QuestionType, difficulty: float):
          # concept_id, not a Concept object: Question only needs to know WHICH concept it
          # belongs to, not the concept's full state -- this also breaks what would otherwise
          # be a circular class reference (Concept -> Question -> Concept).
          self.concept: Concept = concept
          self.question_type: QuestionType = question_type
          self.difficulty: float = difficulty


class Answer:
     def __init__(self, correct: bool):
          self.correct: bool = correct


class QuestionAnswer:
    def __init__(self, question: Question, answer: Answer, time_since_last: float = 0.0):
        self.question: Question = question
        self.answer: Answer = answer
        self.time_since_last_question: float = time_since_last