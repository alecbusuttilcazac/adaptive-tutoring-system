from enum import Enum
import numpy as np
from difflib import SequenceMatcher

from config import NUMERIC_ANSWER_TOLERANCE, FUZZY_MATCH_THRESHOLD


class IncrementalMean:
    def __init__(self):
        self.sum: float = 0.0
        self.count: int = 0

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += value

    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


class Concept:
    def __init__(self, name: str, id: int, explanation: str = "",
                allowed_question_types: set["QuestionType"] | None = None):
        self.name: str = name
        self.id: int = id
        self.questions_by_type: dict["QuestionType", set["Question"]] = {
            question_type: set() for question_type in QuestionType
        }
        self.explanation: str = explanation
        # None means "no restriction" -- every type is semantically valid for this concept.
        self.allowed_question_types: set["QuestionType"] = (
            allowed_question_types if allowed_question_types is not None else set(QuestionType)
        )
        # Must allow at least one type per category, or sample_question_type() could roll a
        # category with zero compatible types for this concept and have nothing to sample.
        for category in QuestionTypeCategory:
            if not (self.allowed_question_types & category.types()):
                raise ValueError(
                    f"Concept {id} ({name!r}) allows no {category.name} question types -- "
                    f"every concept must allow at least one type per category"
                )

    # Compares/hashes by id, not object identity.
    def __eq__(self, other):
        return isinstance(other, Concept) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Area: # A student can have different areas of study.
    def __init__(self, id: int) -> None:
        from contentsequencing.bandit import ZPDBandit

        self.id: int = id
        self.zpd_bandit: ZPDBandit
        self.history: list["QuestionAnswer"] = []
        self.last_history_size: int = 0 # self.history length to stop double batch pos advancement
                # chronological order - fit_MAP's smoothness prior relies on it
        self.theta_by_concept: dict[int, np.ndarray] = {} # concept_id -> fitted theta array
        self.familiarity_by_concept: dict[int, float] = {}
                # concept_id -> self-reported familiarity tier, seeds theta_before before any fit exists
        self.scores_by_type: dict[QuestionType, IncrementalMean] = {
            question_type: IncrementalMean() for question_type in QuestionType
        }  # question_type -> incremental mean
        self.batch_position: int = 0
                # position within the current batch of questions (0 to BATCH_SIZE-1)
        self.delayed_scores: list["QuestionAnswer"] = []
                # (answer, score) pairs for delayed scoring

    # Compares/hashes by id, same reasoning as Concept.__eq__/__hash__ above -- needed
    # for Area to work correctly as a set element (e.g. Student.areas) or dict key.
    def __eq__(self, other):
        return isinstance(other, Area) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def is_unpractised(self, concept_id: int) -> bool:
        return concept_id not in self.theta_by_concept


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
    NUMERIC = 4

    def answer_type(self) -> type:
        match self:
            case QuestionType.TRUE_OR_FALSE:      return bool
            case QuestionType.MULTIPLE_CHOICE:    return int
            case QuestionType.FILL_IN_THE_BLANK:  return str
            case QuestionType.FLASHCARD:          return bool # dependent on user's self-assessment
            case QuestionType.NUMERIC:            return float
            case _: raise ValueError(f"Unknown question type: {self}")


    def guess_floor(self) -> float:
        # Each activity types has a guess_floor parameter (the chance of getting a question
        # right by guessing).
        match self:
            case QuestionType.TRUE_OR_FALSE:      return 0.5
            case QuestionType.MULTIPLE_CHOICE:    return 0.25
            case QuestionType.FILL_IN_THE_BLANK | QuestionType.FLASHCARD | QuestionType.NUMERIC:
                return 0.0
            case _: raise ValueError(f"Unknown question type: {self}")

    def category(self) -> "QuestionTypeCategory":
        if self in QuestionTypeCategory.NON_GENERATIVE.types():
            return QuestionTypeCategory.NON_GENERATIVE
        if self in QuestionTypeCategory.GENERATIVE.types():
            return QuestionTypeCategory.GENERATIVE
        raise ValueError(f"Unknown question type: {self}")


class QuestionTypeCategory(Enum):
    NON_GENERATIVE = 0
    GENERATIVE = 1

    def types(self) -> set:
        if self is QuestionTypeCategory.NON_GENERATIVE:
            return {QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_OR_FALSE}
        return {QuestionType.FILL_IN_THE_BLANK, QuestionType.FLASHCARD, QuestionType.NUMERIC}


class Question:
    def __init__(self, concept: Concept, question_type: QuestionType,
                correct_answer: bool|int|str|float, difficulty: float) -> None:
        self.concept_id: int = concept.id
        self.question_type: QuestionType = question_type
        
        if question_type.answer_type() is not type(correct_answer):
            raise ValueError(f"correct_answer type {type(correct_answer)} does not match "
                    f"question_type {question_type} answer_type {question_type.answer_type()}")
        self.correct_answer: bool | int | str | float = correct_answer
        
        if question_type is QuestionType.FLASHCARD:
            self.correct_answer = True

        self.difficulty: float = difficulty
        concept.questions_by_type[question_type].add(self)
    
    def is_correct(self, answer: bool|int|str|float) -> bool:
        def _fuzzy_match(expected: str, actual: str) -> bool:
            return SequenceMatcher(None, expected.strip().lower(), actual.strip().lower()) \
                    .ratio() >= FUZZY_MATCH_THRESHOLD
        
        if self.question_type.answer_type() is not type(answer):
                raise ValueError(f"correct_answer type {type(answer)} does not match "
                        f"question_type {self.question_type} answer_type "
                        f"{self.question_type.answer_type()}")
        
        if self.question_type is QuestionType.FLASHCARD:
            assert type(answer) is bool
            return answer

        if self.question_type in (QuestionType.TRUE_OR_FALSE, QuestionType.MULTIPLE_CHOICE):
            assert type(answer) is self.question_type.answer_type()
            return answer is self.correct_answer

        if self.question_type is QuestionType.NUMERIC:
            assert type(answer) is float; assert type(self.correct_answer) is float
            return abs(answer - self.correct_answer) < NUMERIC_ANSWER_TOLERANCE
        
        if self.question_type is QuestionType.FILL_IN_THE_BLANK:
            assert type(answer) is str; assert type(self.correct_answer) is str
            return _fuzzy_match(self.correct_answer, answer)

        raise ValueError(f"Unknown question type: {self.question_type}")


# Represents an event in time
class QuestionAnswer:
    def __init__(self, question: Question, correct: bool, time_since_last: float = 0.0) -> None:
        self.question: Question = question
        self.correct: bool = correct
        self.time_since_last_question: float = time_since_last