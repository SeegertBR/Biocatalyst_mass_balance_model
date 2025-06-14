import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns

def sensitivity_and_uncertainty(model, rel_unc=0.05, N=1000):
    param_keys = [
    "Yxs","Yaf","Yxa","Yoa","Yxf",
    "ksmax","kamax","Pamax","kpc","kd","maintenance",
    "k1","k2","Kinhibsat",
    "Ks","Kis","Kia","Ka","Kap","KPrI","KEI","Kpc",
    "kla","O_sat",
    "S0","X0","O0","V0","V_max",
    "F0","Sf","t_start_feed","k_exp",
    "Fi0","If","t_start_inducer_feed",
    "t0", "precursor_batch_mass", "precursor_batch_duration", "number_of_batches"
]
    missing = [k for k in param_keys if not hasattr(model, k)]
    assert not missing, f"Missing params: {missing}"

    # base-case
    t0, C0 = model.solve()
    C0 = np.maximum(C0, 0)
    assert not np.any(np.isnan(C0)) and not np.any(C0<0), "Base solution invalid"

    # sampling
    means  = {k: getattr(model, k) for k in param_keys}
    sigmas = {k: rel_unc * abs(v)       for k,v in means.items()}
    samples= {k: np.clip(np.random.normal(means[k], sigmas[k], N), 0, None)
               for k in param_keys}

    # Monte Carlo
    P_out = np.full(N, np.nan)
    for i in range(N):
        sim = copy.deepcopy(model)
        for k in param_keys:
            setattr(sim, k, samples[k][i])
        try:
            _, C = sim.solve()
            P_out[i] = max(0, C[-1,2])
        except:
            pass
    valid = ~np.isnan(P_out)
    assert valid.any(), "All runs invalid"

    # MC histogram
    plt.hist(P_out[valid], bins=30, edgecolor='k')
    plt.xlabel('Final P'); plt.ylabel('Frequency'); plt.title('MC Uncertainty')
    plt.tight_layout(); plt.show()

    # time‐resolved local sensitivities
    delta = 1e-3
    sens_time = {}
    for k in param_keys:
        orig = getattr(model, k)
        setattr(model, k, orig + delta)
        _, C1 = model.solve()
        setattr(model, k, orig)
        C1 = np.maximum(C1, 0)
        sens_time[k] = (C1[:,2] - C0[:,2]) / delta

    # pick top-10 by maximum absolute sensitivity
    peaks = {k: np.max(np.abs(sens_time[k])) for k in param_keys}
    top10 = sorted(peaks, key=peaks.get, reverse=True)[:10]

    # plot only those ten
    plt.figure(figsize=(10, 6))
    for k in top10:
        plt.plot(t0, sens_time[k], label=k)
    plt.xlabel('Time'); plt.ylabel('∂P/∂param'); plt.title('Top-10 Local Sensitivities Over Time')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.tight_layout(); plt.show()

    # global sensitivities via regression
    df = pd.DataFrame({k: samples[k][valid] for k in param_keys})
    df['P_end'] = P_out[valid]
    reg = LinearRegression().fit(df[param_keys], df['P_end'])
    print('Regression coefficients:')
    for k,c in zip(param_keys, reg.coef_):
        print(f" {k}: {c:.3f}")

    # KDE for all
    thresh = df['P_end'].median()
    high, low = df[df['P_end']>=thresh], df[df['P_end']<thresh]
    for k in param_keys:
        sns.kdeplot(high[k], fill=True, label='high')
        sns.kdeplot(low[k],  fill=True, label='low')
        plt.title(k); plt.tight_layout(); plt.show()
