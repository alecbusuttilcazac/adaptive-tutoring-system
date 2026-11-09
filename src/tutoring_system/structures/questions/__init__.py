from .question import Question
from .generative import GenerativeQuestion, Numeric, FillInTheBlank, Flashcard
from .nongenerative import NonGenerativeQuestion, TrueOrFalse, MultipleChoice, MultipleResponse

# Every question type that can actually be created. Add a new class here when you write one.
CONCRETE_QUESTION_TYPES = (
    Numeric, FillInTheBlank, Flashcard, TrueOrFalse, MultipleChoice, MultipleResponse,
)


def types_of(category: type[Question]) -> set[type[Question]]:
    # The concrete types in a category, e.g. types_of(GenerativeQuestion).
    return {t for t in CONCRETE_QUESTION_TYPES if issubclass(t, category)}


__all__ = ["question", "GenerativeQuestion", "Numeric", "FillInTheBlank", "Flashcard", 
           "NonGenerativeQuestion", "TrueOrFalse", "MultipleChoice", "MultipleResponse",
           "types_of", "CONCRETE_QUESTION_TYPES"]