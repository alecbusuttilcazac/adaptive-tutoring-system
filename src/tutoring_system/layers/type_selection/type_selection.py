import scipy as sp
import numpy as np

from utils.config import config
from tutoring_system.structures.entities import Area, QuestionAnswer, KC
from tutoring_system.structures.questions import (
    Question, GenerativeQuestion, NonGenerativeQuestion, types_of,
)
from tutoring_system.tskirt.tskirt import response_probability


_CATEGORIES = (GenerativeQuestion, NonGenerativeQuestion)

# Early sanity check for normalise_weights(): constraint is n*TYPE_MIN_WEIGHT <= 1 <=
# n*TYPE_MAX_WEIGHT; normalise_weights() re-checks this against its actual input length.
_MAX_CATEGORY_SIZE = max(len(types_of(category)) for category in _CATEGORIES)

if _MAX_CATEGORY_SIZE * config["TYPE_MIN_WEIGHT"] > 1 + config["FLOAT_FEASIBILITY_TOLERANCE"] \
		or _MAX_CATEGORY_SIZE * config["TYPE_MAX_WEIGHT"] < 1 - config["FLOAT_FEASIBILITY_TOLERANCE"]:
	raise ValueError(
		f"TYPE_MAX_WEIGHT ({config['TYPE_MAX_WEIGHT']}) and TYPE_MIN_WEIGHT ({config['TYPE_MIN_WEIGHT']}) are "
		f"infeasible for a category of {_MAX_CATEGORY_SIZE} types: need "
		f"{_MAX_CATEGORY_SIZE} * {config['TYPE_MIN_WEIGHT']} <= 1 <= {_MAX_CATEGORY_SIZE} * {config['TYPE_MAX_WEIGHT']}, "
		f"but got {_MAX_CATEGORY_SIZE * config['TYPE_MIN_WEIGHT']} and {_MAX_CATEGORY_SIZE * config['TYPE_MAX_WEIGHT']}"
	)


def batch_split_index() -> int:
	# Index of the first question in a batch to be scored with theta_after. Read from the config
	# on every call so a sweep can change the batch settings inside one process.
	if config["BATCH_SIZE"] == 1:
		return config["BATCH_SIZE"]  # past the only valid index (0): scored with theta_before
	return round(config["BATCH_SIZE"] * config["BATCH_THETA_SPLIT_FRACTION"])


def _theta_to_category(theta: float) -> type[Question]:
	return NonGenerativeQuestion \
			if np.random.random() > sp.special.expit(config["SIGMOID_STEEPNESS"] * theta) \
			else GenerativeQuestion


def _score_individual_response(area: Area, theta: float, ans: QuestionAnswer) -> None:
	score = ans.correct - response_probability(theta, ans.question.difficulty,
	                                           ans.question.guess_floor())
	area.scores_by_type[type(ans.question)].add(score)


def score_batch_responses(area: Area, theta: float, answers: list[QuestionAnswer],
		clear_delayed: bool = True) -> None:
	for ans in answers:
		_score_individual_response(area, theta, ans)
	if clear_delayed:
		area.delayed_scores.clear()


def _needs_theta_after(batch_position: int) -> bool:
	return batch_position >= batch_split_index()


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
	
	if area.batch_position < config["BATCH_SIZE"] - 1:
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
	return theta <= config["FAMILIARITY_THRESHOLD"] 


# An (almost entirely) AI-generated solution. Normalises weights between the thresholds.
def normalise_weights(weights: list[float]) -> list[float]:
	n = len(weights)
	if n == 0:
		raise ValueError("weights must be non-empty")

	# if n==1, there is only one possible choice
	if n == 1:
		return [1]

	# Recheck feasibility for this call's actual n (module-lvl check only covers _MAX_CATEGORY_SIZE)
	if n * config["TYPE_MIN_WEIGHT"] > 1 + config["FLOAT_FEASIBILITY_TOLERANCE"]:
		raise ValueError(
			f"normalise_weights: TYPE_MIN_WEIGHT ({config['TYPE_MIN_WEIGHT']}) is infeasible for "
			f"{n} weights: {n} * {config['TYPE_MIN_WEIGHT']} = {n * config['TYPE_MIN_WEIGHT']} > 1"
		)
	if n * config["TYPE_MAX_WEIGHT"] < 1 - config["FLOAT_FEASIBILITY_TOLERANCE"]:
		raise ValueError(
			f"normalise_weights: TYPE_MAX_WEIGHT ({config['TYPE_MAX_WEIGHT']}) is infeasible for "
			f"{n} weights: {n} * {config['TYPE_MAX_WEIGHT']} = {n * config['TYPE_MAX_WEIGHT']} < 1"
		)

	# n*min==1 is special-cased for a shortcut/symmetry with the n*max==1 case below.
	# Bisection already reaches this answer on its own. n*max==1 is not optional: float
	# rounding can leave it fractionally under 1, which the doubling loop below can't cross.
	if abs(n * config["TYPE_MIN_WEIGHT"] - 1) <= config["FLOAT_FEASIBILITY_TOLERANCE"]:
		return [config["TYPE_MIN_WEIGHT"]] * n
	if abs(n * config["TYPE_MAX_WEIGHT"] - 1) <= config["FLOAT_FEASIBILITY_TOLERANCE"]:
		return [config["TYPE_MAX_WEIGHT"]] * n

	# convert all weights to +ve, sum to 1
	weights = list(sp.special.softmax([w / config["NORM_SOFTMAX_TEMPERATURE"] for w in weights]))

	# Scale all weights by one factor t, clip to bounds, find t so the clipped sum is 1
	def clipped_sum(t: float) -> float:
		return sum(min(max(w * t, config["TYPE_MIN_WEIGHT"]), config["TYPE_MAX_WEIGHT"]) for w in weights)

	t_low, t_high = 0.0, 1.0
	doublings = 0
	while clipped_sum(t_high) < 1:
		t_high *= 2
		doublings += 1
		if doublings > config["NORM_MAX_DOUBLINGS"]:
			# backstop in case the feasibility check above was bypassed
			raise RuntimeError(f"normalise_weights: t_high search did not converge after "
					f"{config['NORM_MAX_DOUBLINGS']} doublings")

	for _ in range(config["NORM_SCALE_SEARCH_ITERATIONS"]):
		t_mid = (t_low + t_high) / 2
		if clipped_sum(t_mid) < 1:
			t_low = t_mid
		else:
			t_high = t_mid

	return [min(max(w * t_high, config["TYPE_MIN_WEIGHT"]), config["TYPE_MAX_WEIGHT"]) for w in weights]


def sample_question_type(theta: float, area: Area, kc: KC) -> type[Question]:
	# get all the possible types and then get each type's weight (unless mean is unreliable)
	# sorted: a set of classes iterates in a different order in each process, which would make
	# the seeded choice below irreproducible
	possible_types = sorted(types_of(_theta_to_category(theta)) & kc.allowed_question_types,
	                        key=lambda question_class: question_class.__name__)
	if len(possible_types) == 0:
		raise ValueError(f"KC {kc.id} ('{kc.name!r}') does not allow for at least " 
		                 f"one question type in each category.")
	
	weights = [
		area.scores_by_type[question_type].mean()
		if area.scores_by_type[question_type].count >= config["MIN_RESPONSES_FOR_TYPE_SIGNAL"]
		else 0.0
		for question_type in possible_types
	]

	# normalise weights and return a weighted sample
	weights = normalise_weights(weights)
	return possible_types[np.random.choice(len(possible_types), p=weights)]