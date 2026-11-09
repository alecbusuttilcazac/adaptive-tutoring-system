from abc import ABC, abstractmethod


class Question(ABC):
    def __init__(self, kc_id: int, difficulty: float, prompt: str = "") -> None:
        self.kc_id: int = kc_id
        self.difficulty: float = difficulty
        self.prompt = prompt

    @abstractmethod
    def is_correct(self, answer) -> bool: ...

    @abstractmethod
    def guess_floor(self) -> float: ...
