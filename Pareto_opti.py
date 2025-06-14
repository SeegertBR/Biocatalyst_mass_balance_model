import os
import numpy as np
import copy
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from E_model import EcoliFedBatch
from mpl_toolkits.mplot3d import Axes3D



class EColiParetoProblem(ElementwiseProblem):
    def __init__(self, model):
        self.template_model = model

        # Define bounds for your 11 decision variables.
        # Replace these placeholder bounds with actual problem-specific bounds.
        xl = np.array([
            0.01,  # F0 lower bound (flow rate)
            0.1,   # Sf lower bound (feed concentration)
            0.0001,   # k_exp lower bound (growth rate exponent)
            0.0,   # S0 lower bound (initial substrate)
            0.01,  # X0 lower bound (initial biomass)
            0.0,   # precursor_batch_mass lower bound
            0.1,   # precursor_batch_duration lower bound
            1.0,   # t_end lower bound (total time)
            0.0,   # t0 lower bound (first pulse time)
            0.1,   # dt lower bound (interval between pulses)
            1.0    # n_pulses_raw lower bound (number of pulses, real)
        ])

        xu = np.array([
            5.0,   # F0 upper bound
            100, # Sf upper bound
            0.5,   # k_exp upper bound
            10.0,  # S0 upper bound
            2.0,   # X0 upper bound
            40.0,  # precursor_batch_mass upper bound
            10.0,  # precursor_batch_duration upper bound
            72.0,  # t_end upper bound
            24.0,  # t0 upper bound
            10.0,  # dt upper bound
            10.0   # n_pulses_raw upper bound
        ])

        super().__init__(
            n_var=11,
            n_obj=3,
            n_constr=1,  # One constraint
            xl=xl,
            xu=xu,
            elementwise_evaluation=True
        )

    def _evaluate(self, x, out, *args, **kwargs):
        model = copy.deepcopy(self.template_model)

        try:
            (model.F0, model.Sf, model.k_exp, model.S0, model.X0,
             model.precursor_batch_mass, model.precursor_batch_duration,
             model.t_end, t0, dt, n_pulses_raw) = x

            n_pulses = int(np.clip(np.round(n_pulses_raw), 1, 10))
            t_end_pulses = t0 + dt * (n_pulses - 1)

            # Constraint: pulse schedule must fit inside fermentation time
            if t_end_pulses > model.t_end:
                # Penalize objective with large values
                out["F"] = [1e6, 1e6, 1e6]
                # Constraint violation positive means violated
                out["G"] = [t_end_pulses - model.t_end]
                return

            model.precursor_batch_times = [t0 + i * dt for i in range(n_pulses)]

            # Solve the model ODEs
            t, C = model.solve()

            # Safety check on output shape
            if C.shape[1] < 3:
                raise ValueError(f"Unexpected output shape: {C.shape}")

            final_product = C[-1, 2]  # Assuming index 2 corresponds to product conc.
            total_precursor = model.precursor_batch_mass * n_pulses

            if model.k_exp == 0:
                total_substrate = model.F0 * model.Sf * model.t_end + model.S0
            else:
                total_substrate = (model.Sf * model.F0 * (
                    np.exp(model.k_exp * model.t_end) - 1
                ) / model.k_exp) + model.S0

            # Objectives: maximize product (so minimize negative), minimize precursor and substrate
            out["F"] = [-final_product, total_precursor, total_substrate]
            out["G"] = [0.0]  # no constraint violation

        except Exception as e:
            print(f"Evaluation failed: {e}")
            # Penalize heavily if failure
            out["F"] = [1e6, 1e6, 1e6]
            out["G"] = [1e6]



def plot_pareto_front(F):
    plt.ion()  # Enable interactive mode (optional but helpful)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(-F[:, 0], F[:, 1], F[:, 2], s=60, alpha=0.8)

    # Axis labels
    ax.set_xlabel("Final D-DIBOA [g/L]", labelpad=15, fontsize=14)
    ax.set_ylabel("Total Precursor [g]", labelpad=15, fontsize=14)
    ax.set_zlabel("Total Substrate [g]", labelpad=15, fontsize=14)

    # Title
    ax.set_title("Pareto Front: E. coli Fed-Batch Optimization", pad=25, fontsize=18)

    # Camera angle — rotate to show more of right-hand side
    ax.view_init(elev=30, azim=210)

    # Show interactive plot
    fig.tight_layout()
    plt.show()


def run_pareto_optimization(model, generations, pop_size, seed, plot=True):
    problem = EColiParetoProblem(model=model)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', generations),
        seed=seed,
        save_history=True,
        verbose=True,
        n_processes=os.cpu_count()
    )

    if plot:
        plot_pareto_front(res.F)

    return res.F, res.X


def printpareto(model, gen):
    F, X = run_pareto_optimization(
        model,
        generations=gen,
        pop_size=1000,
        seed=42,
        plot=True
    )

    products = -F[:, 0]
    precursors = F[:, 1]
    substrates = F[:, 2]

    idx_maxP = np.argmax(products)
    idx_minPr = np.argmin(precursors)
    idx_minS = np.argmin(substrates)

    # === Find Knee Point ===
    ideal = np.array([products.max(), precursors.min(), substrates.min()])
    points = np.stack([products, precursors, substrates], axis=1)
    distances = np.linalg.norm(points - ideal, axis=1)
    idx_knee = np.argmin(distances)

    def print_solution(title, idx):
        vars_ = X[idx]
        (F0, Sf, k_exp, S0, X0,
         prec_amt, prec_dur,
         t_end, t0, dt, n_pulses_raw) = vars_
        n_pulses = int(np.clip(np.round(n_pulses_raw), 1, 10))

        print(f"\n{title}")
        print(f"  D-DIBOA           = {products[idx]:.4f} g/L")
        print(f"  Total Precursor   = {precursors[idx]:.2f} g")
        print(f"  Total Substrate   = {substrates[idx]:.2f} g")
        print(f"  Number of Pulses  = {n_pulses}")
        print("  Decision Variables:")
        print(f"    F0 (flow rate)           = {F0:.3f}")
        print(f"    Sf (feed conc.)          = {Sf:.3f}")
        print(f"    k_exp (exp. growth rate) = {k_exp:.4f}")
        print(f"    S0 (initial substrate)   = {S0:.3f}")
        print(f"    X0 (initial biomass)     = {X0:.3f}")
        print(f"    precursor_batch_amount   = {prec_amt:.3f}")
        print(f"    precursor_batch_duration = {prec_dur:.3f}")
        print(f"    t_end (total time)       = {t_end:.2f}")
        print(f"    t0 (first pulse time)    = {t0:.2f}")
        print(f"    dt (interval)            = {dt:.2f}")

    print_solution("Best-by-Product Solution:", idx_maxP)
    print_solution("Best-by-Precursor Solution:", idx_minPr)
    print_solution("Best-by-Substrate Solution:", idx_minS)
    print_solution("Knee Point (Best Trade-Off):", idx_knee)


if __name__ == "__main__":
    # Assuming you have an EcoliFedBatch model instance named `model`
    model = EcoliFedBatch()

    print("Running Pareto optimization (this may take some time)...")
    printpareto(model, gen=50)  # 50 generations, can be reduced for quick tests
    print("Done")
