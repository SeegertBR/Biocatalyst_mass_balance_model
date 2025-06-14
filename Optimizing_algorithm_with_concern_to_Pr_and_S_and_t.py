import numpy as np
from E_model import EcoliFedBatch
from scipy.optimize import differential_evolution, minimize
from functools import partial
import copy
model = EcoliFedBatch()
# overwrite with your “optimal” values, but fix the naming:

# safe exponential to avoid overflow
EXP_MAX = 50.0
test_pulse = None
def safe_exp(x):
    try:
        return np.exp(x)
    except OverflowError:
        return np.exp(700) if x > 0 else np.exp(-700)

# scalar objective wrapper (picklable) for multiprocess
def obj_scalar(x, model, n_pulses,
               lambda_precursor=0.0, lambda_substrate=0.0,
               lambda_t=0.0, lambda_inducer=0.0):
    return global_optifunc_Pr_S_t_I(x, model, n_pulses,
                                    lambda_precursor,
                                    lambda_substrate,
                                    lambda_t,
                                    lambda_inducer)[0]


def global_optifunc_Pr_S_t_I(
    x, model, n_pulses,
    lambda_precursor=0.0,
    lambda_substrate=0.0,
    lambda_t=0.0,
    lambda_inducer=0.0
):
    """
    Returns: (objective, final_product, total_precursor, total_substrate, total_inducer, t_end)
    """
    try:
        # 1) instantiate fresh model
        model_instance = copy.deepcopy(model)
        (
            model_instance.F0,
            model_instance.Sf,
            model_instance.k_exp,
            model_instance.S0,
            model_instance.X0,
            model_instance.precursor_batch_mass,
            model_instance.precursor_batch_duration,
            model_instance.t_end,
            model_instance.t0,
            model_instance.dt,
            model_instance.Fi0,
            model_instance.If,
            model_instance.t_start_inducer_feed
        ) = x

        # 2) set up pulses
        batch_times = [model_instance.t0 + i *  model_instance.dt for i in range(n_pulses)]
        model_instance.precursor_batch_times = batch_times
        if any(tp > model_instance.t_end for tp in batch_times):
            return 1e6, 0.0, 0.0, 0.0, 0.0, model_instance.t_end

        # 3) run sim
        t, C = model_instance.solve()
        if np.any(np.isnan(C)) or np.any(C < 0):
            return 1e6, 0.0, 0.0, 0.0, 0.0, model_instance.t_end

        final_product = C[-1, 2]

        # 4) compute total precursor added
        total_precursor = model_instance.precursor_batch_mass * model_instance.precursor_batch_duration * n_pulses

        # 5) compute total substrate and inducer added
        u_sub = np.array([model_instance.feed_rate(ti) for ti in t])
        u_ind = np.array([model_instance.feed_rate_inducer(ti) for ti in t])

        total_substrate = np.trapz(u_sub * model_instance.Sf, t)
        total_inducer  = np.trapz(u_ind * model_instance.If, t)
        
        
        t_end =model_instance.t_end
        # 7) weighted objective
        obj = (
            -final_product+
            lambda_precursor * total_precursor+ 
            lambda_substrate * total_substrate+ 
            lambda_inducer * total_inducer+ 
            lambda_t * t_end
        )
        if final_product < 0.0001:
            return 1e9, final_product, total_precursor, total_substrate, total_inducer, t_end
    
        if model_instance.If < 0.001:
            return 1e9, final_product, total_precursor, total_substrate, total_inducer, t_end
    
        return obj, final_product, total_precursor, total_substrate, total_inducer, t_end

    except Exception as e:
        print(f"Error in simulation with x={x}, n_pulses={n_pulses}: {e}")
        return 1e6, 0.0, 0.0, 0.0, 0.0, 0.0


def optimize_Pr_and_S_t_I(model, generations, pop_size,
                          lambda_precursor=0,
                          lambda_substrate=0,
                          lambda_t=0, 
                          lambda_inducer=0):
    x0 = np.array([
        model.F0,
        model.Sf,
        model.k_exp,
        model.S0,
        model.X0,
        model.precursor_batch_mass,
        model.precursor_batch_duration,
        model.t_end,
        model.t0,
        model.dt,
        model.Fi0,
        model.If,
        model.t_start_inducer_feed
    ])
    bounds = [
        (0.001, 100),
        (1, 500),
        (0.001, 1),
        (0.01, 50),
        (0.001, 10),
        (0.5, 100),
        (0.1, 10),
        (1, 144),
        (1, 20),
        (0.1, 10),
        (0.001, 10),
        (1, 100),
        (0, model.t_end * 0.9)
    ]

    best_obj = np.inf
    best_params = None
    best_pulses = None
    best_out = None

    for n in range(1, 4):
        print(f"Optimizing for {n} pulses...")
        func = partial(obj_scalar, model=model, n_pulses=n,
                       lambda_precursor=lambda_precursor,
                       lambda_substrate=lambda_substrate,
                       lambda_t=lambda_t,
                       lambda_inducer=lambda_inducer)

        res_g = differential_evolution(func, bounds,
                                       strategy='best1bin',
                                       maxiter=generations,
                                       popsize=pop_size,
                                       polish=False,
                                       workers=-1)
        res_l = minimize(func, x0=res_g.x,
                         method='L-BFGS-B', bounds=bounds)
        candidate = res_l if res_l.fun < res_g.fun else res_g

        final_out = global_optifunc_Pr_S_t_I(candidate.x, model, n,
                                              lambda_precursor, lambda_substrate,
                                              lambda_t, lambda_inducer)
        print(f"  -> obj = {final_out[0]:.4f}, product = {final_out[1]:.4f}, prec = {final_out[2]:.2f}, sub = {final_out[3]:.2f}, ind = {final_out[4]:.2f}")

        if final_out[0] < best_obj:
            best_obj = final_out[0]
            best_params = candidate.x
            best_pulses = n
            best_out = final_out

    # build optimized model
    optimized = copy.deepcopy(model)
    labels = ['F0','Sf','k_exp','S0','X0',
              'precursor_batch_mass','precursor_batch_duration',
              't_end','t0','dt','Fi0','If','t_start_inducer_feed']
    for name, val in zip(labels, best_params):
        setattr(optimized, name, val)
    optimized.precursor_batch_times = [best_params[8] + i*best_params[9] for i in range(best_pulses)]

    print("--- Optimal Results ---")
    print(f"Pulses: {best_pulses}")
    for lbl, val in zip(labels, best_params):
        print(f"{lbl}: {val:.4f}")

    print(f"\nFinal Product: {best_out[1]:.4f}\nTotal Precursor: {best_out[2]:.2f}\nTotal Substrate: {best_out[3]:.2f}\nTotal Inducer: {best_out[4]:.2f}\nt_end: {best_out[5]:.2f}")

    print("\nCopy this parameter vector to reuse:")
    formatted_params = ', '.join([f"{v:.6f}" for v in best_params])
    print(f"params = np.array([{formatted_params}])")

    return optimized, best_params, best_pulses, best_out
