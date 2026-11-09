from difflib import SequenceMatcher

from tutoring_system.structures.questions.question import Question
from utils.config import config


class GenerativeQuestion(Question):
    # is_correct stays abstract
    def guess_floor(self) -> float:
        return 0.0


class Numeric(GenerativeQuestion):
    def __init__(self, kc_id: int, difficulty: float, correct_answer: float,
                 prompt: str = "") -> None:
        super().__init__(kc_id, difficulty, prompt)
        self.correct_answer = correct_answer
    
    def is_correct(self, answer) -> bool:
        return abs(answer - self.correct_answer) < config["FLOAT_FEASIBILITY_TOLERANCE"]


class Flashcard(GenerativeQuestion):
    # No need for __init__, since no extra args are required
    def is_correct(self, answer) -> bool:
        return answer # relies on the student's honesty


class FillInTheBlank(GenerativeQuestion):
    def __init__(self, kc_id: int, difficulty: float, correct_answer: str,
                 prompt: str = "") -> None:
        super().__init__(kc_id, difficulty, prompt)
        self.correct_answer = correct_answer

    @staticmethod
    def _fuzzy_match(expected: str, actual: str) -> bool:
        return SequenceMatcher(None, expected.strip().lower(), actual.strip().lower()) \
                .ratio() >= config["FUZZY_MATCH_THRESHOLD"]
    
    def is_correct(self, answer) -> bool:
        return self._fuzzy_match(self.correct_answer, answer)