# Shared tunable constants, consolidated out of their individual modules so every
# empirically-calibrated value (sigma_vs_T, softmax_exploration, etc.) lives in one
# place. Grouped by the module each constant originally belonged to.


# General constants
FLOAT_FEASIBILITY_TOLERANCE = 1e-6  # tolerance for floating-point feasibility checks


# structures/structures.py
NUMERIC_ANSWER_TOLERANCE = 1e-6  # max abs difference for a NUMERIC answer to count as correct
FUZZY_MATCH_THRESHOLD = 0.85  # min similarity for a FILL_IN_THE_BLANK answer to count as correct


# [0] tskirt.py
MIN_DELTA_T = 1e-3
LAMBDA_FORGET = 0.08
THETA_BOUND = 10.0


# [1] contentsequencing/zpd.py
LEVEL_FLEXIBILITY_THRESHOLD = 1


# [1] contentsequencing/bandit.py
REWARD_FUNCTION_NUM_THETAS = 15  # num thetas to be considered in the reward function.
REWARD_FUNCTION_HALF_LIFE = 5  # half life < k
SOFTMAX_TEMPERATURE = 0.065
UCB_SCALE = 0.12  # UCB Multiplier
MASTERY_THETA = 1.25  # minimum theta needed to mark a concept as mastered (completed)
MIN_EXERCISES = 3


# [2] type_selection/type_selection.py
SIGMOID_STEEPNESS = 3.0  # steepness of the sigmoid function
BATCH_SIZE = 4 # number of questions to select before updating the corresponding thetas
BATCH_THETA_SPLIT_FRACTION = 0.5  # fraction of a batch (from its start) assigned theta_before; the 
		# remainder is assigned theta_after, once the batch's refit produces it (delayed)
FAMILIARITY_THRESHOLD = -1  # threshold for a concept to be considered "familiar" (for question type selection)
# Unrelated to FAMILIARITY_THRESHOLD above - these are seed values for theta_before on a
# concept's first-ever batch, picked from the student's self-reported familiarity tier.
FAMILIARITY_UNFAMILIAR = -2.0
FAMILIARITY_SOME = -1.0
FAMILIARITY_CONFIDENT = 0.0
TYPE_MAX_WEIGHT = 0.7  # max weight for question type selection in the scoring function
TYPE_MIN_WEIGHT = 0.12  # min weight for question type selection in the scoring function
# constraint: n*TYPE_MIN_WEIGHT <= 1 <= n*TYPE_MAX_WEIGHT, for the largest category size n
NORM_SOFTMAX_TEMPERATURE = 0.20  # lower sharpens the softmax inside normalise_weights(); 1.0 = no scaling
NORM_SCALE_SEARCH_ITERATIONS = 100  # bisection steps - float64 precision is reached well before this
NORM_MAX_DOUBLINGS = 2000  # backstop against an unbounded t_high search if the feasibility check above is ever bypassed
MIN_RESPONSES_FOR_TYPE_SIGNAL = 7 # minimum number of responses for a type to be considered trusted