import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns

def montecarloplots(modelchosen):
    model = modelchosen

    # Extended list of parameters including inducer kinetics and feed
    param_keys = [
    # Growth & Yield
    "Yxs", "Yaf", "Yxa", "Yoa", "Yxf",
    # Kinetic Rate Constants
    "ksmax", "kamax", "Pamax", "kpc", "kd", "maintenance",
    # Inducer kinetics
    "k1", "k2", "Kinhibsat",
    # Inhibition & Saturation
    "Ks", "Kis", "Kia", "Ka", "Kap", "KPrI", "KEI", "Kpc",
    # Oxygen transfer & C‐compose
    "kla", "O_sat",
    # Initial conditions
    "S0", "X0", "O0", "V_max",
    # Feeding strategy
    "F0", "Sf", "t_start_feed", "k_exp",
    # Inducer feed
    "Fi0", "If", "t_start_inducer_feed",
    # Batch feeding
    "t0", "precursor_batch_mass", "precursor_batch_duration", "number_of_batches"
]

    # Extract means and compute sigmas
    param_means = {k: getattr(model, k) for k in param_keys}
    rel_uncertainty = 0.05
    param_sigmas = {
        k: rel_uncertainty * abs(v) if isinstance(v, (int, float)) else 0
        for k, v in param_means.items()
    }

    num_simulations = 1000
    num_states = 9  # Now tracking 9 states: S, X, P, A, O, Pr, E, I, V
    # Storage for simulation results: simulations × time steps × num_states
    all_C = np.zeros((num_simulations, model.steps, num_states))
    
    # Generate samples
    samples = {
        k: np.random.normal(loc=param_means[k], scale=param_sigmas[k], size=num_simulations)
        for k in param_keys
    }
    
    # Monte Carlo loop
    for i in range(num_simulations):
        sim_model = copy.deepcopy(model)
        for k in param_keys:
            setattr(sim_model, k, samples[k][i])
        t, C = sim_model.solve()
        all_C[i] = C  # C has shape (steps, 9)

    # Variable indices
    variable_indices = {
        "S": 0,
        "X": 1,
        "P": 2,
        "A": 3,
        "O": 4,
        "Pr": 5,
        "E": 6,
        "I": 7,
        "V": 8,
    }

    # Plot uncertainty for each variable
    for var, idx in variable_indices.items():
        plt.figure(figsize=(10, 5))
        for sim in range(num_simulations):
            plt.plot(t, all_C[sim, :, idx], color='blue', alpha=0.01)
        mean_curve = np.mean(all_C[:, :, idx], axis=0)
        perc10 = np.percentile(all_C[:, :, idx], 10, axis=0)
        perc90 = np.percentile(all_C[:, :, idx], 90, axis=0)
        plt.plot(t, mean_curve, 'k-', linewidth=2, label='Mean')
        plt.plot(t, perc10, 'k--', linewidth=2, label='10th percentile')
        plt.plot(t, perc90, 'k--', linewidth=2, label='90th percentile')
        plt.title(f'Uncertainty in {var}')
        plt.xlabel('Time (h)')
        plt.ylabel(f'{var} (g/L)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Time-integrated regression-based sensitivity
    for var, idx in variable_indices.items():
        time_steps = t.size
        src_time = {k: [] for k in param_keys}

        for ti in range(time_steps):
            y = all_C[:, ti, idx]
            df = pd.DataFrame({k: samples[k] for k in param_keys})
            df['Output'] = y
            X = df[param_keys]
            y_vals = df['Output'].values
            reg = LinearRegression().fit(X, y_vals)
            coeffs = pd.Series(reg.coef_, index=param_keys)
            std_x = X.std()
            std_y = np.std(y_vals)
            if std_y > 0:
                src = coeffs * std_x / std_y
            else:
                src = pd.Series(0, index=param_keys)
            for k in param_keys:
                src_time[k].append(src[k])

        # Aggregate
        avg_abs_src = {k: np.mean(np.abs(src_time[k])) for k in param_keys}
        avg_src_series = pd.Series(avg_abs_src)
        ranking = avg_src_series.rank(ascending=False, method='min').astype(int)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(avg_src_series.index, avg_src_series, edgecolor='k', color='lightblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Time-Integrated SRC for {var}')
