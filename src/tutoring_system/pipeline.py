import random
import numpy as np

from tutoring_system.layers.kc_seq.bandit import ZPDBandit
from tutoring_system.structures.entities import Student, Area, KC, QuestionAnswer
from tutoring_system.structures.questions import (
    Numeric, FillInTheBlank, Flashcard, TrueOrFalse, MultipleChoice, MultipleResponse,
)
from utils.config import config
from tutoring_system.tskirt.tskirt import fit_kc, response_probability, responses_for_kc
from tutoring_system.layers.type_selection.type_selection import needs_explanation, process_response, sample_question_type, score_batch_responses


# Placeholder
MOCK_FAMILIARITY_BY_KC = {
    1: config["FAMILIARITY_CONFIDENT"],
    2: config["FAMILIARITY_CONFIDENT"],
    3: config["FAMILIARITY_SOME"],
    4: config["FAMILIARITY_UNFAMILIAR"],
    5: config["FAMILIARITY_UNFAMILIAR"],
}


def build_mock_area(difficulties=(0.0,)) -> tuple[Student, Area]:
	student = Student(id=0)
	area = Area(id=0)
	student.areas.add(area)

	# AI-generated kcs and prerequisites
	c1 = KC(id=1, name="Variables",
		explanation="A variable stores a value under a name.")
	c2 = KC(id=2, name="Loops",
		explanation="A loop repeats a block of code.")
	c3 = KC(id=3, name="Conditionals",
		explanation="An if-statement branches based on a condition.")
	c4 = KC(id=4, name="Functions",
		explanation="A function packages reusable code with inputs/outputs.")
	c5 = KC(id=5, name="Recursion",
		explanation="A function that calls itself to solve smaller sub-problems.")


	for kc in (c1, c2, c3, c4, c5):
		options = ["a", "b", "c", "d"]
		for difficulty in difficulties:
			kc.add_question(MultipleChoice(kc.id, difficulty, options, 0))
			kc.add_question(MultipleResponse(kc.id, difficulty, options, {0, 2}))
			kc.add_question(TrueOrFalse(kc.id, difficulty, True))
			kc.add_question(FillInTheBlank(kc.id, difficulty, "answer"))
			kc.add_question(Flashcard(kc.id, difficulty))
			kc.add_question(Numeric(kc.id, difficulty, 1.0))

	kcs = [
		(c1, 1, set()),              # level 1, no prerequisites
		(c2, 2, {c1}),               # level 2, needs Variables
		(c3, 2, {c1}),               # level 2, needs Variables (parallel branch to Loops)
		(c4, 3, {c2, c3}),           # level 3, needs both Loops and Conditionals
		(c5, 4, {c4}),               # level 4, needs Functions
    ]

	area.zpd_bandit = ZPDBandit(kcs)

	return student, area


def _refit_and_flush(area: Area, kc: KC):
	area.theta_by_kc[kc.id] = fit_kc(area, kc.id)
	theta_after = area.theta_by_kc[kc.id][-1]
	score_batch_responses(area, theta_after, area.delayed_scores, clear_delayed=True)


def run_question(area: Area, kc: KC, student=None, on_response=None):
	# student: a SimulatedStudent that answers (and supplies the time gap); without one the answer
	# is mocked from the system's own theta. on_response(kc, theta_before, answer) is an
	# optional hook for experiments that record what happened.
	trajectory = area.theta_by_kc.get(kc.id)
	if trajectory is not None:
		theta_before = trajectory[-1]
	else:
		theta_before = MOCK_FAMILIARITY_BY_KC.get(kc.id, config["FAMILIARITY_SOME"])

	if needs_explanation(theta_before): # if "needs explanation", we still ask the user first
		print("KC Explanation:", kc.explanation)

	question_type = sample_question_type(theta_before, area, kc)
	question = random.choice(list(kc.questions_by_type[question_type]))

	if student is not None:
		ans = student.respond(question)
	else:
		# Mock answer
		p_correct = response_probability(theta_before, question.difficulty, question.guess_floor())
		correct = np.random.random() < p_correct
		ans = QuestionAnswer(question, correct)
	area.history.append(ans)
	if on_response is not None:
		on_response(kc, theta_before, ans)

	batch_closed = process_response(area, theta_before, ans)
	if batch_closed:
		_refit_and_flush(area, kc)


def run_exercise(area: Area, kc: KC, n_questions: int, student=None, on_response=None):
	for _ in range(n_questions):
		run_question(area, kc, student, on_response)

	# Refit if any of this kc's responses are not in a fit yet. A batch that is longer than
	# the exercise, or that ends part-way through it, would otherwise leave the kc without
	# a fit when update_kc needs one.
	if len(responses_for_kc(area, kc.id)) > len(area.theta_by_kc.get(kc.id, ())):
		_refit_and_flush(area, kc)

	# Once per exercise, after all its questions
	area.zpd_bandit.update_kc(kc, area.theta_by_kc[kc.id])
	


def run_session(area: Area, n_exercises: int, questions_per_exercise: int = 8, student=None,
                on_response=None, on_exercise_end=None):
    for _ in range(n_exercises):
        if not area.zpd_bandit.graph.eligible:
            break  # every reachable kc is mastered
        kc = area.zpd_bandit.select_kc().kc
        run_exercise(area, kc, questions_per_exercise, student, on_response)
        if on_exercise_end is not None:
            on_exercise_end(area, kc)



def setup():
	student, area = build_mock_area()
	kc = area.zpd_bandit.select_kc().kc
	run_question(area, kc)
	