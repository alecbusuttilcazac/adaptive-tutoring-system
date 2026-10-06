# Question/QuestionType test plan:
#   1. Constructing a Question raises if correct_answer's type doesn't match
#      question_type.answer_type() -- catches mismatches early rather than surfacing as
#      an opaque failure later inside is_correct.
#   2. is_correct for MULTIPLE_CHOICE/TRUE_OR_FALSE: exact match is correct, anything else
#      is wrong.
#   3. is_correct for NUMERIC: an answer within NUMERIC_ANSWER_TOLERANCE counts as correct,
#      outside it does not -- exact float equality would be too strict for this type.
#   4. is_correct for FILL_IN_THE_BLANK: fuzzy-matches on minor differences (case,
#      surrounding whitespace, a small typo) but still rejects a genuinely wrong answer.
#   5. is_correct for FLASHCARD is self-graded -- whatever the student claims (True/False)
#      is returned as-is, regardless of what correct_answer was constructed with.

import pytest

from structures.structures import Concept, Question, QuestionType
from config import NUMERIC_ANSWER_TOLERANCE


@pytest.fixture
def concept():
    return Concept("test concept", 1)


def test_constructor_rejects_mismatched_answer_type(concept):
    with pytest.raises(ValueError):
        Question(concept, QuestionType.MULTIPLE_CHOICE, correct_answer="not an int", difficulty=0.0)


def test_multiple_choice_is_correct(concept):
    q = Question(concept, QuestionType.MULTIPLE_CHOICE, correct_answer=2, difficulty=0.0)
    assert q.is_correct(2) is True
    assert q.is_correct(1) is False


def test_true_or_false_is_correct(concept):
    q = Question(concept, QuestionType.TRUE_OR_FALSE, correct_answer=True, difficulty=0.0)
    assert q.is_correct(True) is True
    assert q.is_correct(False) is False


def test_numeric_is_correct_within_tolerance(concept):
    q = Question(concept, QuestionType.NUMERIC, correct_answer=3.14, difficulty=0.0)
    assert q.is_correct(3.14) is True
    assert q.is_correct(3.14 + NUMERIC_ANSWER_TOLERANCE / 2) is True
    assert q.is_correct(3.14 + NUMERIC_ANSWER_TOLERANCE * 10) is False


def test_fill_in_the_blank_fuzzy_match(concept):
    q = Question(concept, QuestionType.FILL_IN_THE_BLANK, correct_answer="photosynthesis", difficulty=0.0)
    assert q.is_correct("photosynthesis") is True
    assert q.is_correct("  PHOTOSYNTHESIS  ") is True
    assert q.is_correct("photosyntesis") is True  # small typo, still close enough
    assert q.is_correct("mitochondria") is False


def test_flashcard_is_self_graded(concept):
    # correct_answer is forced to True at construction (flashcards have no real correct
    # answer), but is_correct should just echo back whatever the student claims.
    q = Question(concept, QuestionType.FLASHCARD, correct_answer=True, difficulty=0.0)
    assert q.is_correct(True) is True
    assert q.is_correct(False) is False
