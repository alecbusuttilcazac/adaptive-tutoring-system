from enum import Enum

import numpy as np


class Concept:
    def __init__(self, name: str, id: int):
        self.name: str = name
        self.id: int = id

    # Compares/hashes by id, not object identity.
    def __eq__(self, other):
        return isinstance(other, Concept) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Area: # A student can have different areas of study.
    def __init__(self, id: int) -> None:
        from contentsequencing.bandit import ZPDBandit

        self.id: int = id
        self.zpdes: ZPDBandit 
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

    # Compares/hashes by id
    def __eq__(self, other):
        return isinstance(other, Student) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class QuestionType(Enum):
      MULTIPLE_CHOICE = 0
      TRUE_OR_FALSE = 1
      FILL_IN_THE_BLANK = 2
      FLASHCARD = 3

      def guess_floor(self) -> float:
            # Each activity types has a guess_floor parameter (the chance of getting a question
            # right by guessing).
            if self == QuestionType.MULTIPLE_CHOICE:   return 0.25
            elif self == QuestionType.TRUE_OR_FALSE:   return 0.5
            elif self == QuestionType.FILL_IN_THE_BLANK:  return 0.0
            elif self == QuestionType.FLASHCARD:    return 0.0
            else: raise ValueError("Unknown question type: {}".format(self))


class Question:
     def __init__(self, concept: Concept, question_type: QuestionType, difficulty: float) -> None:
          self.concept: Concept = concept
          self.question_type: QuestionType = question_type
          self.difficulty: float = difficulty


class Answer:
     def __init__(self, correct: bool) -> None:
          self.correct: bool = correct


class QuestionAnswer:
    def __init__(self, question: Question, answer: Answer, time_since_last: float = 0.0) -> None:
        self.question: Question = question
        self.answer: Answer = answer
        self.time_since_last_question: float = time_since_last