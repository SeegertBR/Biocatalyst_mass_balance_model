import numpy as np
from scipy.integrate import odeint
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
# Import your model



def SALib(model):
    # ---- Instantiate a model to get the time grid ----
    model = model
    t = np.linspace(model.t_start,
                    model.t_end,
                    model.steps)
    n_t = len(t)

 # ── List of parameters and their ±range for sampling ──
    param_keys = [
    "Yxs",
    "Yaf",
    "Yxa",
    "Yoa",
    "Yxf",
    "ksmax",
    "kamax",
    "Pamax",
    "kpc",
    "kd",
    "maintenance",
    "k1","k2","Kinhibsat",
    "Ks","Kis","Kia","Ka","Kap","KPrI","KEI","Kpc",
    "kla","O_sat",
    "S0","X0","V0",
    "F0","Sf","t_start_feed","k_exp",
    "Fi0","If","t_start_inducer_feed",
    "t0", "precursor_batch_mass", "precursor_batch_duration", "number_of_batches"
]

    

    
    bounds = [
        [0.3, 0.6],       # Yxs
        [0.05, 0.2],      # Yaf
        [0.1, 0.3],       # Yxa
        [0.3, 0.6],       # Yoa
        [0.05, 0.2],      # Yxf
    
        [0.9, 1.5],       # ksmax
        [0.1, 1.0],       # kamax
        [0.1, 1.0],       # Pamax
        [0.1, 0.5],       # kpc
        [0.0001, 0.001],     # kd
        [0.005, 0.02],    # maintenance
    
        [0.1, 2],      # k1
        [0.0001, 0.01],      # k2
        [10, 30],         # Kinhibsat
    
        [0.03, 0.5],       # Ks
        [1, 20],       # Kis
        [1, 10],       # Kia
        [0.01, 0.1],       # Ka
        [0.5, 1],       # Kap
        [0.01, 1],    # KPrI
        [10, 30],    # KEI
        [0.001, 0.1],       # Kpc
        [0.01, 0.1],       # Kinducer inhib
    
        [50, 300],       # kla
        [0.001, 0.01],       # O_sat
    
        [1, 20],  # S0
        [0.1, 2],       # X0
    
        [0.001, 1],       # F0
        [10, 200],          # Sf
        [1, 5],      # t_start_feed
        [0.001, 0.01],      # k_exp
    
        [0.0001, 0.1],       # Fi0
        [5, 20],          # If
        [1.0, 5.0],        # t_start_inducer_feed
        [5.0, 15.0],#"t0", 
        [1.0, 40.0],#"precursor_batch_mass", 
        [1.0, 5.0],#"precursor_batch_duration", 
        [1.0, 8.0]#"number_of_batches"
        
        
    ]

    problem = {
        'num_vars': len(param_keys),
        'names': param_keys,
        'bounds': bounds
    }

    # ── Model runner: returns the full P(t) vector ──

    def evaluate_model_time(params):
        m = copy.deepcopy(model)
        for name, val in zip(param_keys, params):
            setattr(m, name, val)
        _, sol = m.solve()
        return sol[:, 2]  # P(t)

    # Generate Sobol samples (second-order enabled!)
    param_values = saltelli.sample(problem, 1024, calc_second_order=True)
    n_samples = param_values.shape[0]

    n_jobs = multiprocessing.cpu_count()
    P_list = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_model_time)(X) for X in tqdm(param_values)
    )
    P_matrix = np.vstack(P_list)  # (n_samples, n_t)

    # First- and second-order index containers
    S1_time = np.full((len(param_keys), n_t), np.nan)
    S2_time = np.full((len(param_keys), len(param_keys), n_t), np.nan)

    # Time-dependent Sobol analysis
    for i in range(n_t):
        Y_i = P_matrix[:, i]
        if np.allclose(Y_i, Y_i[0]):
            continue
        Si = sobol.analyze(problem,
                           Y_i,
                           calc_second_order=True,
                           print_to_console=False)
        S1_time[:, i] = Si['S1']
        for j in range(len(param_keys)):
            for k in range(j + 1, len(param_keys)):
                S2_time[j, k, i] = Si['S2'][j, k]

    # Plot Top 5 S1
    top5 = np.argsort(np.nanmax(S1_time, axis=1))[-5:]
    plt.figure(figsize=(10, 6))
    for idx in top5:
        plt.plot(t, S1_time[idx], label=param_keys[idx])
    plt.xlabel("Time (h)")
    plt.ylabel("Sobol S₁")
    plt.title("Time‐dependent First‐order Sobol Indices for P(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: Print or plot selected S2 interactions
    # Example: Print top 5 S2 interactions at final time point
    t_index = -1  # Final time point
    S2_final = S2_time[:, :, t_index]
    triu_indices = np.triu_indices(len(param_keys), k=1)
    S2_pairs = [
        (param_keys[i], param_keys[j], S2_final[i, j])
        for i, j in zip(*triu_indices)
        if not np.isnan(S2_final[i, j])
    ]
    S2_pairs_sorted = sorted(S2_pairs, key=lambda x: -x[2])[:5]
    print("\nTop 5 Second-order Interactions (final time point):")
    for a, b, s2 in S2_pairs_sorted:
        print(f"{a} × {b} : S₂ = {s2:.4f}")
        
    avg_S2 = np.nanmean(S2_time, axis=2)  # shape: (param, param)
    interaction_strengths = []
    for i in range(len(param_keys)):
        for j in range(i + 1, len(param_keys)):
            if not np.isnan(avg_S2[i, j]):
                interaction_strengths.append((param_keys[i], param_keys[j], avg_S2[i, j], i, j))
    
    # Sort and take top 5
    top5_s2 = sorted(interaction_strengths, key=lambda x: -x[2])[:5]
    
    # Plot
    plt.figure(figsize=(10, 6))
    for a, b, _, i, j in top5_s2:
        plt.plot(t, S2_time[i, j], label=f"{a} × {b}")
    plt.xlabel("Time (h)")
    plt.ylabel("Sobol S₂")
    plt.title("Top 5 Second-order Sobol Interactions Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()