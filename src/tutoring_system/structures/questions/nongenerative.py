from abc import abstractmethod
import random

from tutoring_system.structures.questions.question import Question
from utils.config import config

#to add: question type, correct answer (bool|int|str|float|set[int]), possible_answers
class NonGenerativeQuestion(Question):
    # guess_floor and is_correct stay abstract
    pass


class TrueOrFalse(NonGenerativeQuestion):
    def __init__(self, kc_id: int, difficulty: float, correct_answer: bool, 
                 prompt: str = "") -> None:
        super().__init__(kc_id, difficulty, prompt)
        self.correct_answer = correct_answer

    def is_correct(self, answer: bool) -> bool:
        return answer == self.correct_answer

    def guess_floor(self) -> float:
        return 0.5            


class MultipleResponse(NonGenerativeQuestion):
    def __init__(self, kc_id: int, difficulty: float, possible_answers: list[str],
                 correct_answers: set[int], prompt: str = "") -> None:
        super().__init__(kc_id, difficulty, prompt)
        self.possible_answers = possible_answers

        order = list(range(len(possible_answers)))
        random.shuffle(order)
        self.correct_answers = frozenset(order.index(i) for i in correct_answers)
        self.possible_answers = tuple(possible_answers[i] for i in order)

    def is_correct(self, answer: set[int]) -> bool:
        return answer == self.correct_answers

    def guess_floor(self) -> float:
        # All/some/no options can be chosen
        return 0.5 ** len(self.possible_answers)  # The same as 1 / (2 ** len(possible_answers))


class MultipleChoice(NonGenerativeQuestion):
    def __init__(self, kc_id: int, difficulty: float, possible_answers: list[str],
                 correct_answer: int, prompt: str = "") -> None:
        super().__init__(kc_id, difficulty, prompt)
        self.possible_answers = possible_answers
        
        order = list(range(len(possible_answers)))
        random.shuffle(order)
        self.correct_answer = order.index(correct_answer)
        self.possible_answers = [possible_answers[i] for i in order]

    def is_correct(self, answer: int) -> bool:
        return answer == self.correct_answer

    def guess_floor(self) -> float:
        return 1 / len(self.possible_answers)