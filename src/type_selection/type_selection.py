import scipy as sp
import numpy as np

from config import FLOAT_FEASIBILITY_TOLERANCE
from structures.structures import Area, QuestionType, QuestionTypeCategory, QuestionAnswer, Concept
from tskirt.tskirt import response_probability
from config import SIGMOID_STEEPNESS, BATCH_SIZE, BATCH_THETA_SPLIT_FRACTION, \
		FAMILIARITY_THRESHOLD, TYPE_MAX_WEIGHT, TYPE_MIN_WEIGHT, \
		NORM_SCALE_SEARCH_ITERATIONS, NORM_MAX_DOUBLINGS, NORM_SOFTMAX_TEMPERATURE, \
		MIN_RESPONSES_FOR_TYPE_SIGNAL


# Early sanity check for normalise_weights(): constraint is n*TYPE_MIN_WEIGHT <= 1 <=
# n*TYPE_MAX_WEIGHT; normalise_weights() re-checks this against its actual input length.
_MAX_CATEGORY_SIZE = max(len(category.types()) for category in QuestionTypeCategory)

if _MAX_CATEGORY_SIZE * TYPE_MIN_WEIGHT > 1 + FLOAT_FEASIBILITY_TOLERANCE \
		or _MAX_CATEGORY_SIZE * TYPE_MAX_WEIGHT < 1 - FLOAT_FEASIBILITY_TOLERANCE:
	raise ValueError(
		f"TYPE_MAX_WEIGHT ({TYPE_MAX_WEIGHT}) and TYPE_MIN_WEIGHT ({TYPE_MIN_WEIGHT}) are "
		f"infeasible for a category of {_MAX_CATEGORY_SIZE} types: need "
		f"{_MAX_CATEGORY_SIZE} * {TYPE_MIN_WEIGHT} <= 1 <= {_MAX_CATEGORY_SIZE} * {TYPE_MAX_WEIGHT}, "
		f"but got {_MAX_CATEGORY_SIZE * TYPE_MIN_WEIGHT} and {_MAX_CATEGORY_SIZE * TYPE_MAX_WEIGHT}"
	)


if BATCH_SIZE == 1:
	BATCH_SPLIT_INDEX = BATCH_SIZE 
			# past the only valid index (0) - the question is assigned to theta_before
else:
	BATCH_SPLIT_INDEX = round(BATCH_SIZE * BATCH_THETA_SPLIT_FRACTION) 
			# index of the first question in a batch to be assigned theta_after


def _theta_to_category(theta: float) -> QuestionTypeCategory:
	return QuestionTypeCategory.NON_GENERATIVE \
			if np.random.random() > sp.special.expit(SIGMOID_STEEPNESS * theta) \
			else QuestionTypeCategory.GENERATIVE


def _score_individual_response(area: Area, theta: float, ans: QuestionAnswer) -> None:
	score = ans.correct - response_probability(theta, ans.question.difficulty,
	                                           ans.question.question_type.guess_floor())
	area.scores_by_type[ans.question.question_type].add(score)


def score_batch_responses(area: Area, theta: float, answers: list[QuestionAnswer],
		clear_delayed: bool = True) -> None:
	for ans in answers:
		_score_individual_response(area, theta, ans)
	if clear_delayed:
		area.delayed_scores.clear()


def _needs_theta_after(batch_position: int) -> bool:
	return batch_position >= BATCH_SPLIT_INDEX


def _advance_batch_position(area: Area) -> bool:
	if area.last_history_size != len(area.history) - 1:
		# must be called exactly once per new response; a double-call or a skipped
		# call would silently desync batch_position from the actual history length
		raise ValueError(
			f"advance_batch_position called with area.history length {len(area.history)}, expected "
			f"exactly one more than last_history_size ({area.last_history_size}) -- called twice "
			f"for the same response, or skipped for one or more responses"
		)
	area.last_history_size = len(area.history)
	
	if area.batch_position < BATCH_SIZE - 1:
		area.batch_position += 1
		return False
	else:
		area.batch_position = 0
		return True


def process_response(area: Area, theta_before: float, ans: QuestionAnswer) -> bool:
	if _needs_theta_after(area.batch_position):
		area.delayed_scores.append(ans)
	else:
		_score_individual_response(area, theta_before, ans)

	return _advance_batch_position(area)


def needs_explanation(theta: float) -> bool:
	return theta <= FAMILIARITY_THRESHOLD 


# An (almost entirely) AI-generated solution. Normalises weights between the thresholds.
def normalise_weights(weights: list[float]) -> list[float]:
	n = len(weights)
	if n == 0:
		raise ValueError("weights must be non-empty")

	# if n==1, there is only one possible choice
	if n == 1:
		return [1]

	# Recheck feasibility for this call's actual n (module-lvl check only covers _MAX_CATEGORY_SIZE)
	if n * TYPE_MIN_WEIGHT > 1 + FLOAT_FEASIBILITY_TOLERANCE:
		raise ValueError(
			f"normalise_weights: TYPE_MIN_WEIGHT ({TYPE_MIN_WEIGHT}) is infeasible for "
			f"{n} weights: {n} * {TYPE_MIN_WEIGHT} = {n * TYPE_MIN_WEIGHT} > 1"
		)
	if n * TYPE_MAX_WEIGHT < 1 - FLOAT_FEASIBILITY_TOLERANCE:
		raise ValueError(
			f"normalise_weights: TYPE_MAX_WEIGHT ({TYPE_MAX_WEIGHT}) is infeasible for "
			f"{n} weights: {n} * {TYPE_MAX_WEIGHT} = {n * TYPE_MAX_WEIGHT} < 1"
		)

	# n*min==1 is special-cased for a shortcut/symmetry with the n*max==1 case below.
	# Bisection already reaches this answer on its own. n*max==1 is not optional: float
	# rounding can leave it fractionally under 1, which the doubling loop below can't cross.
	if abs(n * TYPE_MIN_WEIGHT - 1) <= FLOAT_FEASIBILITY_TOLERANCE:
		return [TYPE_MIN_WEIGHT] * n
	if abs(n * TYPE_MAX_WEIGHT - 1) <= FLOAT_FEASIBILITY_TOLERANCE:
		return [TYPE_MAX_WEIGHT] * n

	# convert all weights to +ve, sum to 1
	weights = list(sp.special.softmax([w / NORM_SOFTMAX_TEMPERATURE for w in weights]))

	# Scale all weights by one factor t, clip to bounds, find t so the clipped sum is 1
	def clipped_sum(t: float) -> float:
		return sum(min(max(w * t, TYPE_MIN_WEIGHT), TYPE_MAX_WEIGHT) for w in weights)

	t_low, t_high = 0.0, 1.0
	doublings = 0
	while clipped_sum(t_high) < 1:
		t_high *= 2
		doublings += 1
		if doublings > NORM_MAX_DOUBLINGS:
			# backstop in case the feasibility check above was bypassed
			raise RuntimeError(f"normalise_weights: t_high search did not converge after "
					f"{NORM_MAX_DOUBLINGS} doublings")

	for _ in range(NORM_SCALE_SEARCH_ITERATIONS):
		t_mid = (t_low + t_high) / 2
		if clipped_sum(t_mid) < 1:
			t_low = t_mid
		else:
			t_high = t_mid

	return [min(max(w * t_high, TYPE_MIN_WEIGHT), TYPE_MAX_WEIGHT) for w in weights]


def sample_question_type(theta: float, area: Area, concept: Concept) -> QuestionType:
	# get all the possible types and then get each type's weight (unless mean is unreliable)
	possible_types = list(_theta_to_category(theta).types() & concept.allowed_question_types)
	if len(possible_types) == 0:
		raise ValueError(f"Concept {concept.id} ('{concept.name!r}') does not allow for at least " 
		                 f"one question type in each category.")
	
	weights = [
		area.scores_by_type[question_type].mean()
		if area.scores_by_type[question_type].count >= MIN_RESPONSES_FOR_TYPE_SIGNAL
		else 0.0
		for question_type in possible_types
	]

	# normalise weights and return a weighted sample
	weights = normalise_weights(weights)
	return possible_types[np.random.choice(len(possible_types), p=weights)]