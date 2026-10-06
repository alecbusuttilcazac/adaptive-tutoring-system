import random
import numpy as np

from contentsequencing.bandit import ZPDBandit
from structures.structures import Student, Area, Concept, Question, QuestionType, QuestionAnswer
from config import FAMILIARITY_CONFIDENT, FAMILIARITY_SOME, FAMILIARITY_UNFAMILIAR
from tskirt.tskirt import fit_concept, response_probability
from type_selection.type_selection import needs_explanation, process_response, sample_question_type, score_batch_responses


# Placeholder until real self-reported familiarity collection exists -- concept_id ->
# familiarity tier, used only to seed theta_before before a concept's first fit exists.
MOCK_FAMILIARITY_BY_CONCEPT = {
    1: FAMILIARITY_CONFIDENT,
    2: FAMILIARITY_CONFIDENT,
    3: FAMILIARITY_SOME,
    4: FAMILIARITY_UNFAMILIAR,
    5: FAMILIARITY_UNFAMILIAR,
}


def build_mock_area() -> tuple[Student, Area]:
	student = Student(id=0)
	area = Area(id=0)
	student.areas.add(area)

	# AI-generated concepts and prerequisites
	c1 = Concept(id=1, name="Variables",
		explanation="A variable stores a value under a name.")
	c2 = Concept(id=2, name="Loops",
		explanation="A loop repeats a block of code.")
	c3 = Concept(id=3, name="Conditionals",
		explanation="An if-statement branches based on a condition.")
	c4 = Concept(id=4, name="Functions",
		explanation="A function packages reusable code with inputs/outputs.")
	c5 = Concept(id=5, name="Recursion",
		explanation="A function that calls itself to solve smaller sub-problems.")


	for concept in (c1, c2, c3, c4, c5):
		Question(concept, QuestionType.MULTIPLE_CHOICE, correct_answer=0, difficulty=0.0)
		Question(concept, QuestionType.TRUE_OR_FALSE, correct_answer=True, difficulty=0.0)
		Question(concept, QuestionType.FILL_IN_THE_BLANK, correct_answer="answer", difficulty=0.0)
		Question(concept, QuestionType.FLASHCARD, correct_answer=True, difficulty=0.0)
		Question(concept, QuestionType.NUMERIC, correct_answer=1.0, difficulty=0.0)

	concepts = [
		(c1, 1, set()),              # level 1, no prerequisites
		(c2, 2, {c1}),               # level 2, needs Variables
		(c3, 2, {c1}),               # level 2, needs Variables (parallel branch to Loops)
		(c4, 3, {c2, c3}),           # level 3, needs both Loops and Conditionals
		(c5, 4, {c4}),               # level 4, needs Functions
    ]

	area.zpd_bandit = ZPDBandit(concepts)

	return student, area


def _refit_and_flush(area: Area, concept: Concept):
	area.theta_by_concept[concept.id] = fit_concept(area, concept.id)
	theta_after = area.theta_by_concept[concept.id][-1]
	score_batch_responses(area, theta_after, area.delayed_scores, clear_delayed=True)
	area.zpd_bandit.update_concept(concept, area.theta_by_concept[concept.id])


def run_question(area: Area, concept: Concept):
	trajectory = area.theta_by_concept.get(concept.id)
	if trajectory is not None:
		theta_before = trajectory[-1]
	else:
		theta_before = MOCK_FAMILIARITY_BY_CONCEPT.get(concept.id, FAMILIARITY_SOME)

	if needs_explanation(theta_before): # if "needs explanation", we still ask the user first
		print("Concept Explanation:", concept.explanation)

	question_type = sample_question_type(theta_before, area, concept)
	question = random.choice(list(concept.questions_by_type[question_type]))

	# Mock answer
	p_correct = response_probability(theta_before, question.difficulty, question_type.guess_floor())
	correct = np.random.random() < p_correct
	ans = QuestionAnswer(question, correct)
	area.history.append(ans)

	batch_closed = process_response(area, theta_before, ans)
	if batch_closed:
		_refit_and_flush(area, concept)


def run_exercise(area: Area, concept: Concept, n_questions: int):
	for _ in range(n_questions):
		run_question(area, concept)
		
	if len(area.delayed_scores) > 0:
		_refit_and_flush(area, concept)
	


def run_session(area: Area, n_exercises: int):
    for _ in range(n_exercises):
        concept = area.zpd_bandit.select_concept().concept
        run_exercise(area, concept, n_questions=8)



def setup():
	student, area = build_mock_area()
	concept = area.zpd_bandit.select_concept().concept
	run_question(area, concept)
	