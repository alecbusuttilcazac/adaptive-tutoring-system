import numpy as np

from tutoring_system.structures.questions import (
    Question, GenerativeQuestion, NonGenerativeQuestion, CONCRETE_QUESTION_TYPES, types_of,
)
from utils.stats import IncrementalMean


class KC:
    def __init__(self, name: str, id: int, explanation: str = "",
                allowed_question_types: set[type[Question]] | None = None):
        self.name: str = name
        self.id: int = id
        self.questions_by_type: dict[type[Question], list[Question]] = {
            question_class: [] for question_class in CONCRETE_QUESTION_TYPES
        }
        self.explanation: str = explanation
        # None means "no restriction" -- every type is semantically valid for this kc.
        self.allowed_question_types: set[type[Question]] = (
            allowed_question_types if allowed_question_types is not None
            else set(CONCRETE_QUESTION_TYPES)
        )
        # Must allow at least one type per category, or sample_question_type() could roll a
        # category with zero compatible types for this kc and have nothing to sample.
        for category in (GenerativeQuestion, NonGenerativeQuestion):
            if not (self.allowed_question_types & types_of(category)):
                raise ValueError(
                    f"kc {id} ({name!r}) allows no {category.__name__} types -- "
                    f"every kc must allow at least one type per category"
                )

    def add_question(self, question: Question) -> Question:
        # A question only knows its kc's id, so the kc does the registering.
        if question.kc_id != self.id:
            raise ValueError(f"question belongs to kc {question.kc_id}, "
                             f"not kc {self.id}")
        self.questions_by_type[type(question)].append(question)
        return question

    # Compares/hashes by id, not object identity.
    def __eq__(self, other):
        return isinstance(other, KC) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Area: # A student can have different areas of study.
    def __init__(self, id: int) -> None:
        from tutoring_system.layers.kc_seq.bandit import ZPDBandit

        self.id: int = id
        self.zpd_bandit: ZPDBandit
        self.history: list["QuestionAnswer"] = []
        self.last_history_size: int = 0 # self.history length to stop double batch pos advancement
                # chronological order - fit_MAP's smoothness prior relies on it
        self.theta_by_kc: dict[int, np.ndarray] = {} # kc_id -> fitted theta array
        self.familiarity_by_kc: dict[int, float] = {}
                # kc_id -> self-reported familiarity tier, seeds theta_before before any fit exists
        self.scores_by_type: dict[type[Question], IncrementalMean] = {
            question_class: IncrementalMean() for question_class in CONCRETE_QUESTION_TYPES
        }  # question class -> incremental mean
        self.batch_position: int = 0
                # position within the current batch of questions (0 to BATCH_SIZE-1)
        self.delayed_scores: list["QuestionAnswer"] = []
                # (answer, score) pairs for delayed scoring

    # Compares/hashes by id, same reasoning as kc.__eq__/__hash__ above -- needed
    # for Area to work correctly as a set element (e.g. Student.areas) or dict key.
    def __eq__(self, other):
        return isinstance(other, Area) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def is_unpractised(self, kc_id: int) -> bool:
        return kc_id not in self.theta_by_kc


class Student:
    def __init__(self, id: int):
        self.id: int = id
        self.areas: set[Area] = set() # A student can have different areas of study.

    # Compares/hashes by id
    def __eq__(self, other):
        return isinstance(other, Student) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


# Represents an event in time
class QuestionAnswer:
    def __init__(self, question: Question, correct: bool, time_since_last: float = 0.0) -> None:
        self.question: Question = question
        self.correct: bool = correct
        self.time_since_last_question: float = time_since_last