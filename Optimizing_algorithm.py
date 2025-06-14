import numpy as np
from E_model import EcoliFedBatch
from scipy.optimize import differential_evolution, minimize
from functools import partial


def global_optifunc(x, model, n_pulses):
    try:
        # Create a new model
        model_instance = model

        # Update model parameters
        model_instance.F0 = x[0]
        model_instance.Sf = x[1]
        model_instance.k_exp = x[2]
        model_instance.S0 = x[3]
        model_instance.X0 = x[4]
        model_instance.precursor_batch_amount = x[5]
        model_instance.precursor_batch_duration = x[6]
        model_instance.t_end = x[7]

        # Precursor pulse times
        t0 = x[8]
        dt = x[9]
        batch_times = [t0 + i * dt for i in range(n_pulses)]
        model_instance.precursor_batch_times = batch_times

        # Run model
        t, C = model_instance.solve();

        final_product = C[-1, 2]
        return -final_product

    #stackoverflow said that this term helps stuff not break, which makes sense
    except Exception as e:
        print(f"Error in simulation with x = {x} and n_pulses = {n_pulses}: {e}")
        return 1e6



def optimize(model):
    # The main optimization routine using a hybrid global–local approach.
    def optimizing_routine(model):
        # Initial guess taken from the given model.instance.
        x0 = np.array([
            model.F0,                   # Feed flow rate (L/h)
            model.Sf,                   # Feed substrate concentration (g/L)
            model.k_exp,                # Feed ramp rate (1/h)
            model.S0,                   # Initial substrate concentration (g/L)
            model.X0,                   # Initial biomass concentration (g/L)
            model.precursor_batch_amount,      # Precursor batch amount (g/L)
            model.precursor_batch_duration,    # Precursor batch duration (h)
            model.t_end,                # End time of simulation (h)
            model.t0,                   # Pulse start time (h)
            model.dt                    # Interval between pulses (h)
        ])
    
        # Define bounds for each parameter to ensure physical constraints make sense. Need to talk with Carina about these
        bounds = [
            (0.01, 10),       # F0 [L/h]
            (1, 150),        # Sf [g/L]
            (0.001, 1),       # k_exp [1/h]
            (0.1, 10),       # S0 [g/L]
            (0.001, 10),      # X0 [g/L]
            (0.5, 50),       # Precursor batch amount [g/L]
            (0.1, 20),        # Precursor batch duration [h]
            (1, 72),          # t_end [h]
            (1, 10),          # t0: Pulse start time [h]
            (0.1, 20)         # dt: Interval between pulses [h]
        ]
    
        # number of pulses to try.
        pulse_candidates = [1, 2, 3]
        best_overall_obj = np.inf
        best_overall_params = None
        best_pulse_count = None
   

    
        # Loop over different pulse counts.
        for n_pulses in pulse_candidates:
            print(f"\nOptimizing for {n_pulses} pulse(s):")
    
            # Global optimization using differential evolution.
            func = partial(global_optifunc, model=model, n_pulses=n_pulses)
            result_global = differential_evolution(
                func=func,
                bounds=bounds,
                strategy='best1bin',
                maxiter=100,
                popsize=25,
                polish=False,
                workers=-1 
        )
            
    
            best_local_obj = np.inf
            best_local_x = None
    
            # Local refinement using L-BFGS-B. Doing both should be better
            result_local = minimize(
                fun=func,
                x0=result_global.x,
                method='L-BFGS-B',
                bounds=bounds
                )
                
    
            # Choose the best result from global and local search. Need to test if it make sense to use time on local runs
            if result_global.fun < result_local:
                best_result = result_global
            else:
                best_result = type('Result', (object,), {'fun': best_local_obj, 'x': best_local_x})
    
            # Defines the highest values
            if best_result.fun < best_overall_obj:
                best_overall_obj = best_result.fun
                best_overall_params = best_result.x
                best_pulse_count = n_pulses
    
        # Print the best overall optimized parameters.
        print("\nBest overall optimized parameters:")
        print(f"Number of pulses: {best_pulse_count}")
        print(f"F0 (Feed flow rate): {best_overall_params[0]:.4f} L/h")
        print(f"Sf (Feed substrate concentration): {best_overall_params[1]:.4f} g/L")
        print(f"k_exp (Feed ramp rate): {best_overall_params[2]:.4f} 1/h")
        print(f"S0 (Initial substrate concentration): {best_overall_params[3]:.4f} g/L")
        print(f"X0 (Initial biomass concentration): {best_overall_params[4]:.4f} g/L")
        print(f"Precursor batch amount: {best_overall_params[5]:.4f} g/L")
        print(f"Precursor batch duration: {best_overall_params[6]:.4f} h")
        print(f"t_end (Simulation end time): {best_overall_params[7]:.4f} h")
        print(f"Pulse start time (t0): {best_overall_params[8]:.4f} h")
        print(f"Interval between pulses (dt): {best_overall_params[9]:.4f} h")
        print(f"Maximum D-DIBOA: {-best_overall_obj:.4f} g/L")
    
    # Calling inner optimization routine.
    optimizing_routine(model)

