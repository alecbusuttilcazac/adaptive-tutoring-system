# Finds the relationship between T (number of responses in a student's history) and the
# best-fitting sigma for that T, since the best sigma was observed to shift with T rather
# than being a single universal constant (see conversation/dissertation notes).
#
# For each candidate T, use golden-section search to find the sigma that
# minimizes mean MSE (true vs fitted theta) over many simulated runs, capping each T's
# search ceiling at the previous T's winner (best sigma only ever decreases as T grows),
# then fit a curve through the resulting (T, best_sigma) pairs.
#
# Parallelized across runs within each sigma evaluation via multiprocessing, since each
# run is fully independent (see fit_all_concepts/simulate_qas -- same independence
# argument applies here).


import numpy as np
from multiprocessing import Pool

import os

import scipy as sp

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from structures.structures import Concept, Answer, Question, QuestionAnswer, QuestionType
from tskirt.tskirt import fit_MAP


def simulate_qas(theta_true, rng, gaps=None, lambda_forget=0.0):
    # theta_true:     the ground-truth proficiency trajectory we're trying to recover.
    # rng:            a numpy random Generator, passed in for reproducibility
    # gaps:           optional array of elapsed time before each response. Defaults to all-1.0 
    #                   gaps, matching the spacing SIGMA was calibrated against.
    # lambda_forget:  if nonzero, decay is applied to theta once per response, scaled by that 
    #                   response's own gap. If lambda_forget=0.0 (the default),
    #                   decayed_theta == theta_true.
    if gaps is None:
        gaps = np.ones(len(theta_true))

    concept = Concept(name="simulated_concept", id=0)
    qas = []

    decayed_theta = np.empty(len(theta_true))
    decayed_theta[0] = theta_true[0]
    for i in range(1, len(theta_true)):
        natural_step = theta_true[i] - theta_true[i - 1]
        decayed_theta[i] = decayed_theta[i - 1] * np.exp(-lambda_forget * gaps[i]) + natural_step

    for i, gap in enumerate(gaps):
        theta_t = decayed_theta[i]

        q_type = rng.choice(list(QuestionType))
        difficulty = rng.uniform(-2, 2)
        question = Question(concept, q_type, difficulty)

        c = q_type.guess_floor()
        p = c + (1 - c) * sp.special.expit(theta_t - difficulty)

        correct = rng.random() < p

        qas.append(QuestionAnswer(question, Answer(correct), time_since_last=float(gap)))
    return qas, decayed_theta


def _single_run_mse(args):
    T, sigma, run = args
    theta_true = np.linspace(-2, 2, T)
    rng = np.random.default_rng(run)
    qas, _ = simulate_qas(theta_true, rng)
    result = fit_MAP(qas, sigma=sigma)
    return np.mean((result.x - theta_true) ** 2)


def _mean_mse_at_sigma(T, sigma, n_runs, pool):
    jobs = [(T, sigma, run) for run in range(n_runs)]
    mses = pool.map(_single_run_mse, jobs)
    return np.mean(mses)


def best_sigma_for_T(T, sigma_min, sigma_max, n_runs, pool, tol=1e-3, max_iters=25):
    # Golden-section search: finds the minimum of mean-MSE-vs-sigma over [sigma_min,
    # sigma_max] without needing a derivative (mean MSE comes from averaging many random
    # simulated fits, so there's no clean formula to differentiate) and without scanning
    # every candidate the way a grid search would. Relies on the function being roughly
    # unimodal (one dip) over the search interval, which matches every sigma-vs-MSE curve
    # collected so far in this project -- typically converges in ~15-25 evaluations
    # regardless of how wide [sigma_min, sigma_max] is.
    invphi = (np.sqrt(5) - 1) / 2  # 1/golden ratio, ~0.618

    a, b = sigma_min, sigma_max
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    f_c = _mean_mse_at_sigma(T, c, n_runs, pool)
    f_d = _mean_mse_at_sigma(T, d, n_runs, pool)

    for _ in range(max_iters):
        if abs(b - a) < tol:
            break
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - invphi * (b - a)
            f_c = _mean_mse_at_sigma(T, c, n_runs, pool)
        else:
            a, c, f_c = c, d, f_d
            d = a + invphi * (b - a)
            f_d = _mean_mse_at_sigma(T, d, n_runs, pool)

    best_sigma = (a + b) / 2
    return best_sigma, min(f_c, f_d)


if __name__ == "__main__":
    T_values = np.array([6, 7, 8, 9, 10, 25, 40, 55, 70, 85, 100, 125, 150, 175, 200, 225, 250, 275, 300, 333, 367, 400, 433, 467, 500, 533, 567, 600, 633, 667, 700, 750, 800, 850, 900, 950, 1000])
    n_runs = 43
    sigma_min = 0.03  # fixed floor for every T -- golden-section search narrows in on its
                       # own, so this doesn't need to shrink like the old grid-scan ceiling did
    initial_sigma_max = 2.0  # upper bound only for the very first T; later Ts use the previous best instead

    results = []
    sigma_max = initial_sigma_max
    with Pool() as pool:
        for T in T_values:
            # Best sigma only ever decreases as T grows (see the combined empirical data
            # gathered across prior runs), so each T's search ceiling is capped at the
            # previous T's winner -- narrows the search interval as we go, encoding that
            # prior knowledge directly rather than re-searching the same wide range every
            # time. Golden-section search (not a grid scan) finds the minimum within
            # [sigma_min, sigma_max] in ~15-25 evaluations regardless of interval width.
            print(f"T={T:<4d}  searching sigma in [{sigma_min:.4f}, {sigma_max:.4f}]", end=", ", flush=True)
            # mean_mse (2nd return value) still computed by best_sigma_for_T -- useful as a
            # sanity check (a smoothly decreasing value as T grows confirms the search found
            # a real minimum) -- just no longer printed here to keep the console log terser.
            best_sigma, best_mse = best_sigma_for_T(int(T), sigma_min, sigma_max, n_runs, pool)
            print(f"best_sigma={best_sigma:.4f}")
            results.append((T, best_sigma))
            sigma_max = best_sigma

    T_arr = np.array([r[0] for r in results], dtype=float)
    sigma_arr = np.array([r[1] for r in results], dtype=float)

    def r_squared(y, pred):
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    fits = {}

    # Logarithmic: sigma = a*log(T) + b
    c = np.polyfit(np.log(T_arr), sigma_arr, deg=1)
    predict = lambda T, c=c: c[0] * np.log(T) + c[1]
    fits["log"] = (predict, r_squared(sigma_arr, predict(T_arr)), f"sigma = {c[0]:.5f}*log(T) + {c[1]:.5f}")

    # Power law: sigma = a * T^b -- fit via log-log linearization
    c = np.polyfit(np.log(T_arr), np.log(sigma_arr), deg=1)
    a, b = np.exp(c[1]), c[0]
    predict = lambda T, a=a, b=b: a * T ** b
    fits["power"] = (predict, r_squared(sigma_arr, predict(T_arr)), f"sigma = {a:.5f} * T^{b:.5f}")


    # Reciprocal: sigma = a/T + b
    c = np.polyfit(1 / T_arr, sigma_arr, deg=1)
    predict = lambda T, c=c: c[0] / T + c[1]
    fits["reciprocal"] = (predict, r_squared(sigma_arr, predict(T_arr)), f"sigma = {c[0]:.5f}/T + {c[1]:.5f}")

    print()
    for name, (_, r2_val, formula) in sorted(fits.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:22s} R^2={r2_val:.5f}   {formula}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(T_arr, sigma_arr, label="empirical best sigma", zorder=5, color="black")
    T_smooth = np.linspace(T_arr.min(), T_arr.max(), 300)
    for name, (predict, r2_val, _) in fits.items():
        # Each predict closure already has its fitted parameters baked in (see above),
        # so plotting is just evaluating it on a smooth T grid -- no re-fitting needed.
        plt.plot(T_smooth, predict(T_smooth), linestyle="--", label=f"{name} (R^2={r2_val:.3f})")

    plt.xlabel("T (no. of responses)")
    plt.ylabel("best sigma")
    plt.title("Relationship between T and best-fitting sigma")
    plt.legend()
    plt.savefig("curves.png")
    print("\nPlot saved to curves.png")
