MIN_DELTA_T = 1e-3
LAMBDA_FORGET = 0.08
THETA_BOUND = 10.0


import numpy as np
import scipy as sp
from enum import Enum
        

class QuestionType(Enum):
	MULTIPLE_CHOICE = 0
	TRUE_OR_FALSE = 1
	FILL_IN_THE_BLANK = 2
	FLASHCARD = 3

	def guess_floor(self):
		# Each activity types has a guess_floor parameter (the chance of getting a question
		# right by guessing). In the 3PL-style IRT formula used in `neg_log_likelihood`.
		if self == QuestionType.MULTIPLE_CHOICE: 		return 0.25
		elif self == QuestionType.TRUE_OR_FALSE: 		return 0.5
		elif self == QuestionType.FILL_IN_THE_BLANK: 	return 0.0
		elif self == QuestionType.FLASHCARD: 			return 0.0
		else: raise ValueError("Unknown question type: {}".format(self))


all_questions: dict[int, dict[QuestionType, list["Question"]]] = {}  # all possible questions


class Concept:
    def __init__(self, name: str, id: int):
        self.name = name
        self.id = id


class Question:
	def __init__(self, concept_id: int, question_type: QuestionType, difficulty: float):
		# concept_id, not a Concept object: Question only needs to know WHICH concept it
		# belongs to, not the concept's full state -- this also breaks what would otherwise
		# be a circular class reference (Concept -> Question -> Concept).
		self.concept_id = concept_id
		self.question_type = question_type
		self.difficulty = difficulty


class Answer:
	def __init__(self, correct: bool):
		self.correct = correct


class QuestionAnswer:
    def __init__(self, question: Question, answer: Answer, time_since_last: float = 0.0):
        self.question = question
        self.answer = answer
        self.time_since_last_question = time_since_last


class Student:
    def __init__(self, id: int):
        self.id = id
        self.history: list[QuestionAnswer] = []
        self.theta_by_concept: dict[int, np.ndarray] = {}  # concept_id -> fitted theta array


def sigma_by_T(T: int) -> float:
    # Observed relationship between T (number of responses) and the best-fitting
    # sigma for that T, based on the results of experiments/sigma_vs_T.py.
    # The power law can never return a negative sigma.
    # R^2 = 0.99833 (error ~0.0017) for 5 < T <= 1000, n_runs = 43
    
    if T <= 5:
        return 1.5  # default sigma for very small T (T=6 -> sigma~1.44)
    else:
        return 5.52289 * (T ** -0.72580)  # power-law fit from sigma_vs_T.py


def responses_for_concept(student: Student, concept_id: int) -> list[QuestionAnswer]:
    # Return the student's history filtered to only responses for the given concept.
    return [qa for qa in student.history if qa.question.concept_id == concept_id]


def fit_all_concepts(student: Student, sigma=None, lambda_forget=LAMBDA_FORGET) -> dict[int, np.ndarray]:
    # Concepts are independent (see neg_log_prior/[[project_tskirt_model]]: no
    # cross-concept correlation term), so this is just fit_MAP looped once per
    # concept -- no joint optimization needed. Pure function, like
    # responses_for_concept: returns a new dict rather than mutating student
    # in place, so the caller decides whether/how to store it (e.g.
    # student.theta_by_concept = fit_all_concepts(student)).
    concept_ids = {qa.question.concept_id for qa in student.history}

    theta_by_concept = {}
    for concept_id in concept_ids:
        qas = responses_for_concept(student, concept_id)
        result = fit_MAP(qas, sigma=sigma, lambda_forget=lambda_forget)
        theta_by_concept[concept_id] = result.x

    return theta_by_concept


def qas_to_arrays(qas: list[QuestionAnswer]) -> tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                      np.ndarray]:
    # Convert a list of QuestionAnswer objects into multiple aligned numpy arrays.
    return( 
        np.array([qa.answer.correct for qa in qas]),
        np.array([qa.question.difficulty for qa in qas]),
        np.array([qa.question.question_type.guess_floor() for qa in qas]),
        np.array([qa.time_since_last_question for qa in qas])
    )


def bernoulli_logli(trues, probs, avg=False):
	# Total Bernoulli log-likelihood.
    # Log-space avoids underflow from multiplying many probabilities.
	if trues.shape != probs.shape: # they must be of the same size
		raise ValueError("Incompatible array shapes")

	falses = np.logical_not(trues) # negate the trues
	
	log_li = (np.sum(np.log(probs[trues])) # sum of log(p) over true entries
            + np.sum(np.log(1.0 - probs[falses]))) # plus log(1-p) over false entries

	if avg:
		return log_li / trues.size
	else:
		return log_li


def neg_log_likelihood(theta, correct, difficulties, guess_floors):
    # 3-PL IRT: p = c + (1-c) * sigmoid(theta - difficulty), where:
	#   p                    Probability of correct response
    #   c                    guess floor of each response
	#   sp.special.expit(.)  sigmoid function
	#   theta                student proficiency at the time of each response
    #   difficulty           difficulty of each response
	# The idea is to stretch the sigmoid up to the guess floor.
    p = guess_floors + (1 - guess_floors) * sp.special.expit(theta - difficulties)
	
    # Clipped to avoid log(0).
    p = np.clip(p, 1e-16, 1 - 1e-16)

    # probabilities give log-likelihood <= 0; negating keeps 0 = best fit
    return -bernoulli_logli(correct, p)


def neg_log_prior(theta, sigma, times_since_last, lambda_forget=0.0):
	# Wiener process prior: theta_t ~ N(theta_{t-1}, sigma^2), where:
	#   theta_t    proficiency at response t
	#   theta_t-1  proficiency at the previous response
	#   sigma      how much proficiency is allowed to drift between responses
	# The idea is to penalize big jumps between consecutive theta estimates so the
	#   trajectory drifts smoothly instead of chasing noise in individual responses.
    # Floored well above machine epsilon: delta_t sits directly under a squared residual as
    #   a divisor, so a near-zero floor (e.g. 1e-16) still blows the term up to ~1e16 for
    #   back-to-back/instant responses -- MIN_DELTA_T is small enough not to distort any real
    #   gap but large enough to keep near-instant responses numerically well-behaved.
    delta_t = np.maximum(times_since_last[1:], MIN_DELTA_T)
    decay = np.exp(-lambda_forget * delta_t)
    decayed_mean = theta[:-1] * decay
    diffs_of_thetas = (theta[1:] - decayed_mean) ** 2
    
	# Sum of squared consecutive jumps, scaled by the Gaussian log-density factor
	#   1/(2*sigma^2); normalizing constant dropped since it doesn't depend on theta.
	# -ve log-prior: same 0-is-best convention as neg_log_likelihood
    return np.sum(diffs_of_thetas / delta_t) / (2 * sigma ** 2)


def neg_log_posterior(theta, correct, difficulties, guess_floors, sigma, times_since_last, 
                      lambda_forget=0.0):
    # Full MAP objective: -log(likelihood * prior) = neg log-likelihood + neg log-prior.
    # Passed directly to scipy.optimize.minimize, which searches over theta.
	# -log(likelihood * prior) = neg log(likelihood) + neg log(prior):
	#   neg_log_likelihood:  how badly does this theta fail to explain the observed responses?
    #   neg_log_prior:       how badly does this theta violate the smoothness prior?
    # -ve overall: MAP wants to *maximize* likelihood*prior, but scipy.optimize.minimize
    #   only minimizes -- minimizing the negation is equivalent to maximizing the original.
    return neg_log_likelihood(theta, correct, difficulties, guess_floors) \
			+ neg_log_prior(theta, sigma, times_since_last, lambda_forget)


def neg_log_posterior_grad(theta, correct, difficulties, guess_floors, sigma, times_since_last, 
                           lambda_forget=0.0):
    # Analytic gradient of neg_log_posterior wrt theta. Without this, scipy estimates the gradient 
    #   via finite differences -- ~T+1 extra objective evaluations per optimizer step. Supplying it 
    #   directly cuts fit_MAP from ~3000+ evaluations to ~T, a ~40x speedup measured on a single
    #   fit (T=50), with identical results (checked against scipy.optimize.check_grad, error ~1e-6).
	# Mostly generated using AI due to more complex logic.

    # gradient of the likelihood term:
    sig = sp.special.expit(theta - difficulties)
    p = guess_floors + (1 - guess_floors) * sig
    p = np.clip(p, 1e-16, 1 - 1e-16)
    # d(sigmoid)/dtheta = sig*(1-sig); chain rule through p = c + (1-c)*sigmoid(theta-difficulty)
    dp_dtheta = (1 - guess_floors) * sig * (1 - sig)
    # d(-logli)/dtheta = -(x/p - (1-x)/(1-p)) * dp/dtheta, elementwise per response
    grad_ll = -(correct / p - (1 - correct) / (1 - p)) * dp_dtheta

    # gradient of the Wiener-process prior term, with forgetting:
    #   neg_log_prior = sum(r_t^2 / (2*sigma^2*dt_t)), where r_t = theta_t - theta_{t-1}*decay_t
    #   and decay_t = exp(-lambda_forget*dt_t). Each interior theta_t appears in two terms:
    #   once as the "current" value in its own term (d/dtheta_t = r_t/(sigma^2*dt_t)), and once
    #   inside the decayed mean of the *next* term (d/dtheta_{t-1} = -r_t*decay_t/(sigma^2*dt_t)).
    delta_t = np.maximum(times_since_last[1:], MIN_DELTA_T)
    decay = np.exp(-lambda_forget * delta_t)
    residual = theta[1:] - theta[:-1] * decay

    grad_prior = np.zeros_like(theta)
    grad_prior[1:] += residual / (sigma ** 2 * delta_t)
    grad_prior[:-1] -= residual * decay / (sigma ** 2 * delta_t)

    return grad_ll + grad_prior


def fit_MAP(qas, sigma=None, theta0=None, lambda_forget=0.0):
    correct, difficulties, guess_floors, times_since_last = qas_to_arrays(qas)
    # Neutral starting guess (average proficiency) if the caller doesn't supply one.
    if theta0 is None:
        theta0 = np.zeros(len(qas))
    # Empirically-tuned default (see sigma_by_T / experiments/sigma_vs_T.py) if the
    # caller doesn't supply one -- callers that need to test/override a specific value
    # (e.g. the sigma_vs_T experiment itself) still can by passing sigma explicitly.
    if sigma is None:
        sigma = sigma_by_T(len(qas)) # where len(qas) is the number of question-answer events.
    # sp.optimize.minimize(objective, x0, jac, args, method), where:
    #   neg_log_posterior       objective being minimized; scipy calls this repeatedly
    #   theta0                  x0, the starting guess -- also the value scipy is allowed to vary
    #   neg_log_posterior_grad  jac, the analytic gradient -- avoids scipy estimating it
    #                             numerically (~40x fewer function evaluations per fit)
    #   L-BFGS-B                method, a quasi-Newton optimizer suited to smooth, continuous,
    #                             moderate-dimensional problems like this one
    result = sp.optimize.minimize(
        neg_log_posterior,
        theta0,
        jac=neg_log_posterior_grad,
        args=(correct, difficulties, guess_floors, sigma, times_since_last, lambda_forget),
        method="L-BFGS-B",
        bounds=[(-THETA_BOUND, THETA_BOUND)] * len(theta0)
    )
    return result  # full OptimizeResult, not just .x -- lets caller check .success/.message too