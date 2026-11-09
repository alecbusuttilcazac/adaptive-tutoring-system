import numpy as np
from scipy.special import expit

from tutoring_system.structures.entities import QuestionAnswer
from tutoring_system.structures.questions import CONCRETE_QUESTION_TYPES, GenerativeQuestion, Question
from utils.config import config


class SimulatedStudent:
    # A student with a hidden true proficiency per kc, so an estimator can be scored
    # against the truth. It answers with the same 3PL model T-SKIRT assumes, learns with every
    # response (faster on generative questions, saturating towards a cap), drifts back toward its
    # starting level between practices, and is better at some question types than others.
    # It draws from its own generator, so runs stay reproducible whatever else uses random.

    def __init__(self, rng: np.random.Generator, initial_theta: dict[int, float]) -> None:
        self.rng = rng
        self.baseline: dict[int, float] = dict(initial_theta)
        self.theta: dict[int, float] = dict(initial_theta)  # true proficiency right now
        self.theta_history: dict[int, list[float]] = {cid: [] for cid in initial_theta}
        # theta_history[c][i] is the true theta the student answered the i-th response to c with
        self.type_offset: dict[type[Question], float] = {
            question_class: float(rng.normal(0.0, config["SIM_TYPE_SKILL_SD"]))
            for question_class in CONCRETE_QUESTION_TYPES
        }
        self.clock: float = 0.0
        self._last_practised: dict[int, float] = {}

    @classmethod
    def from_familiarity(cls, rng: np.random.Generator,
                         familiarity_by_kc: dict[int, float]) -> "SimulatedStudent":
        # Self-reported familiarity is a noisy view of the true starting proficiency.
        noise = config["SIM_FAMILIARITY_NOISE_SD"]
        return cls(rng, {cid: float(tier + rng.normal(0.0, noise))
                         for cid, tier in familiarity_by_kc.items()})

    def respond(self, question: Question) -> QuestionAnswer:
        cid = question.kc_id
        duration = config["SIM_RESPONSE_DURATION"]
        self.clock += duration
        gap = self.clock - self._last_practised.get(cid, self.clock - duration)
        self._last_practised[cid] = self.clock

        base = self.baseline[cid]
        self.theta[cid] = base + (self.theta[cid] - base) * np.exp(-config["SIM_FORGET_RATE"] * gap)
        theta = self.theta[cid]
        self.theta_history[cid].append(theta)

        floor = question.guess_floor()
        p_correct = floor + (1 - floor) * expit(
            theta + self.type_offset[type(question)] - question.difficulty)
        correct = bool(self.rng.random() < p_correct)

        bonus = 1 + config["SIM_GENERATION_BONUS"] * isinstance(question, GenerativeQuestion)
        room = max(0.0, 1.0 - theta / config["SIM_THETA_CAP"])
        self.theta[cid] = theta + config["SIM_LEARNING_RATE"] * bonus * room

        return QuestionAnswer(question, correct, time_since_last=gap)
