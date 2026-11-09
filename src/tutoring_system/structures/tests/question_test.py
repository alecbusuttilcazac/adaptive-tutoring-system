# Question hierarchy test plan:
#   1. The abstract bases (Question, GenerativeQuestion, NonGenerativeQuestion) cannot be
#      instantiated -- every concrete type must implement is_correct and guess_floor.
#   2. is_correct per type: MultipleChoice / TrueOrFalse exact match; MultipleResponse exact
#      set match (including the empty and the full set); Numeric within
#      FLOAT_FEASIBILITY_TOLERANCE; FillInTheBlank fuzzy (case, whitespace, small typo) but not a
#      wrong answer; Flashcard echoes the student's self-assessment.
#   3. guess_floor per type: generative types 0.0, TrueOrFalse 0.5, MultipleChoice 1/n,
#      MultipleResponse 1/2^n.
#   4. Constructing a MultipleChoice / MultipleResponse shuffles the options but the answer key
#      keeps pointing at the same option text.
#   5. The registry: types_of() splits the six concrete types into their categories, and
#      KC.add_question files a question under its own class and rejects another kc's.

import pytest

from tutoring_system.structures.entities import KC
from tutoring_system.structures.questions import (
    Question, GenerativeQuestion, NonGenerativeQuestion, Numeric, FillInTheBlank, Flashcard,
    TrueOrFalse, MultipleChoice, MultipleResponse, CONCRETE_QUESTION_TYPES, types_of,
)
from utils.config import config


OPTIONS = ["Paris", "Rome", "Oslo", "Bern"]


@pytest.fixture
def kc():
    return KC("test kc", 1)


@pytest.mark.parametrize("abstract_class", [Question, GenerativeQuestion, NonGenerativeQuestion])
def test_abstract_bases_cannot_be_instantiated(abstract_class):
    with pytest.raises(TypeError):
        abstract_class(1, 0.0)


def test_multiple_choice_is_correct():
    q = MultipleChoice(1, 0.0, OPTIONS, 2)
    assert q.is_correct(q.correct_answer) is True
    assert q.is_correct((q.correct_answer + 1) % 4) is False


def test_true_or_false_is_correct():
    q = TrueOrFalse(1, 0.0, True)
    assert q.is_correct(True) is True
    assert q.is_correct(False) is False


@pytest.mark.parametrize("key", [set(), {0, 2}, {0, 1, 2, 3}])
def test_multiple_response_is_exact_set_match(key):
    q = MultipleResponse(1, 0.0, OPTIONS, key)
    assert q.is_correct(set(q.correct_answers)) is True
    assert q.is_correct(set(q.correct_answers) | {99}) is False


def test_numeric_is_correct_within_tolerance():
    q = Numeric(1, 0.0, 3.14)
    assert q.is_correct(3.14) is True
    assert q.is_correct(3.14 + config["FLOAT_FEASIBILITY_TOLERANCE"] / 2) is True
    assert q.is_correct(3.14 + config["FLOAT_FEASIBILITY_TOLERANCE"] * 10) is False


def test_fill_in_the_blank_fuzzy_match():
    q = FillInTheBlank(1, 0.0, "photosynthesis")
    assert q.is_correct("photosynthesis") is True
    assert q.is_correct("  PHOTOSYNTHESIS  ") is True
    assert q.is_correct("photosyntesis") is True  # small typo, still close enough
    assert q.is_correct("mitochondria") is False


def test_flashcard_is_self_graded():
    q = Flashcard(1, 0.0)
    assert q.is_correct(True) is True
    assert q.is_correct(False) is False


def test_guess_floors():
    assert Numeric(1, 0.0, 1.0).guess_floor() == 0.0
    assert FillInTheBlank(1, 0.0, "x").guess_floor() == 0.0
    assert Flashcard(1, 0.0).guess_floor() == 0.0
    assert TrueOrFalse(1, 0.0, True).guess_floor() == 0.5
    assert MultipleChoice(1, 0.0, OPTIONS, 0).guess_floor() == pytest.approx(1 / 4)
    assert MultipleResponse(1, 0.0, OPTIONS, {0}).guess_floor() == pytest.approx(1 / 16)


@pytest.mark.parametrize("_", range(50))
def test_multiple_choice_shuffle_keeps_the_key_on_the_same_option(_):
    q = MultipleChoice(1, 0.0, OPTIONS, 0)
    assert q.possible_answers[q.correct_answer] == "Paris"
    assert sorted(q.possible_answers) == sorted(OPTIONS)


@pytest.mark.parametrize("key", [set(), {0, 2}, {0, 1, 2, 3}])
@pytest.mark.parametrize("_", range(25))
def test_multiple_response_shuffle_keeps_the_key_on_the_same_options(key, _):
    q = MultipleResponse(1, 0.0, OPTIONS, key)
    assert {q.possible_answers[i] for i in q.correct_answers} == {OPTIONS[i] for i in key}


def test_types_of_splits_the_concrete_types_by_category():
    assert set(CONCRETE_QUESTION_TYPES) == types_of(GenerativeQuestion) | types_of(NonGenerativeQuestion)
    assert types_of(GenerativeQuestion) == {Numeric, FillInTheBlank, Flashcard}
    assert types_of(NonGenerativeQuestion) == {TrueOrFalse, MultipleChoice, MultipleResponse}


def test_add_question_registers_under_its_class(kc):
    q = kc.add_question(Numeric(kc.id, 0.0, 3.14))
    assert q in kc.questions_by_type[Numeric]
    assert kc.questions_by_type[MultipleChoice] == []


def test_add_question_rejects_another_kcs_question(kc):
    with pytest.raises(ValueError):
        kc.add_question(Numeric(kc.id + 1, 0.0, 3.14))
