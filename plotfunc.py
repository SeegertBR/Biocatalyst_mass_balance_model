import matplotlib.pyplot as plt
import numpy as np



def plot(t, C):
    """
    Overlay plots of all main concentrations (excluding volume).
    """
    labels = [
        'Substrate (g/L)', 'Biomass (g/L)', 'D-DIBOA (g/L)', 'Acetate (g/L)',
        'Oxygen (g/L)', 'Precursor (g/L)', 'Ethanol (g/L)'
    ]
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2'
    ]
    
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(t, C[:, i], label=label, color=colors[i], linewidth=2)
    
    plt.xlabel('Time (h)', fontsize=14)
    plt.ylabel('Concentration (g/L)', fontsize=14)
    plt.title('Fed-Batch E. coli Growth and Product Formation', fontsize=16, fontweight='bold')
    plt.ylim(0, None)  # start y-axis at 0
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_acetate(t, C):
    """
    Plots Acetate concentration over time.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(t, C[:, 3], label='Acetate (g/L)', color='tab:red', linewidth=2)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Acetate (g/L)', fontsize=12)
    plt.title('Acetate Concentration Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_precursor_DDIBOA(t, C):
    """
    Plots Precursor and D-DIBOA concentration over time.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(t, C[:, 5], label='Precursor (g/L)', color='tab:blue', linewidth=2)
    plt.plot(t, C[:, 2], label='D-DIBOA (g/L)', color='tab:green', linewidth=2)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Concentration (g/L)', fontsize=12)
    plt.title('Precursor & D-DIBOA Concentration Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_simulation(t, C):
    """
    Three-panel view:
      1) Substrate, Biomass, Volume
      2) D-DIBOA, Precursor, Inducer
      3) Acetate, Oxygen
    """
    # 1. Substrate, Biomass, Volume
    plt.figure(figsize=(6, 4))
    plt.plot(t, C[:, 0], label='Substrate (g/L)', linewidth=2)
    plt.plot(t, C[:, 1], label='Biomass (g/L)', linewidth=2)
    plt.plot(t, C[:, 8], label='Volume (L)', linewidth=2)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Conc. / Vol.', fontsize=12)
    plt.title('Substrate, Biomass & Volume', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 2. D-DIBOA, Precursor, Inducer
    plt.figure(figsize=(6, 4))
    plt.plot(t, C[:, 2], label='D-DIBOA (g/L)', linewidth=2)
    plt.plot(t, C[:, 5], label='Precursor (g/L)', linewidth=2)
    plt.plot(t, C[:, 7], label='Inducer (g/L)', linewidth=2)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Concentration (g/L)', fontsize=12)
    plt.title('D-DIBOA, Precursor & Inducer', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 3. Acetate, Oxygen
    plt.figure(figsize=(6, 4))
    plt.plot(t, C[:, 3], label='Acetate (g/L)', linewidth=2)
    plt.plot(t, C[:, 4], label='Oxygen (g/L)', linewidth=2)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylabel('Concentration (g/L)', fontsize=12)
    plt.title('Acetate & Oxygen', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_substrate_and_biomass(t, y):
    S = y[:, 0]
    X = y[:, 1]
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, S, label='Substrate (S)', color='tab:blue')
    plt.plot(t, X, label='Biomass (X)', color='tab:green')
    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (g/L)")
    plt.title("Substrate and Biomass vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_biomass_only(t, y):
    X = y[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(t, X, label='Biomass (X)', color='tab:green')
    plt.xlabel("Time (h)")
    plt.ylabel("Biomass (g/L)")
    plt.title("Biomass vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_product_per_biomass(t, y, MW_P=180.16):
    """
    Plot Product per Biomass in mmol D-DIBOA per g Biomass over time.

    Parameters:
        t: time array
        y: simulation result (shape: n_times × n_states)
        MW_P: molar mass of product (g/mol)
    """
    P = y[:, 2]  # g/L product
    X = y[:, 1]  # g/L biomass

    with np.errstate(divide='ignore', invalid='ignore'):
        P_mmol_per_X = np.where(X > 0, (P / MW_P) * 1000 / X, 0)  # mmol/g

    plt.figure(figsize=(8, 5))
    plt.plot(t, P_mmol_per_X, label='Product per Biomass (mmol/g)', color='tab:purple')
    plt.xlabel("Time (h)")
    plt.ylabel("P/X (mmol D-DIBOA / g Biomass)")
    plt.title("Product per Biomass vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


def plot_biotransformation_yield(model, t, y,
                                  MW_P=165.15,
                                  MW_Pr=225.20,
                                  initial_precursor_conc=None):
    """
    Biotransformation Yield:
    BY(t) = mol_P(t) / mol_Pr,i * 100
    where:
      mol_P(t) = [P](t) * V(t) / MW_P
      mol_Pr,i = initial precursor mass (g) / MW_Pr
    """
    P = y[:, 2]   # g/L
    V = y[:, 8]   # L

    # moles of product formed over time
    mol_P = P * V / MW_P

    # total precursor mass added = mass per pulse × number of pulses
    total_precursor_mass = model.precursor_batch_mass * model.number_of_batches

    # moles of precursor initially added
    mol_Pr_i = total_precursor_mass / MW_Pr

    # Calculate BY (yield)
    BY = np.where(mol_Pr_i > 0, 100 * mol_P / mol_Pr_i, 0)
    BY = np.clip(BY, 0, 100)  # Limit to 0-100% for safety

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(t, BY, color='tab:green')
    plt.xlabel('Time (h)')
    plt.ylabel('Biotransformation Yield (%)')
    plt.title('Mole-Based Biotransformation Yield Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Final Biotransformation Yield: {BY[-1]:.1f}%")
# Plot: Specific Productivity (SP)


def plot_specific_productivity(model, t, y, MW_P=149.15):
    """
    Plot instantaneous specific productivity qP(t) in mmol product / gCDW / h.

    Parameters:
      model: your EcoliFedBatch instance (not actually used here, but kept for consistency)
      t:     time vector (h)
      y:     solution matrix (n × states), with
             y[:,2] = P (g/L product), y[:,1] = X (g/L biomass), y[:,8] = V (L)
      MW_P:  molar mass of D-DIBOA (g/mol)
    """
    P = y[:,2]   # g/L
    X = y[:,1]   # g/L
    V = y[:,8]   # L

    # total mmol of product in reactor
    mmol_P = (P * V / MW_P) * 1e3   # mmol

    # instantaneous formation rate in mmol/h
    dmmol_P_dt = np.gradient(mmol_P, t)

    # total biomass in reactor
    X_tot = X * V   # g

    # specific productivity in mmol / g / h
    with np.errstate(divide='ignore', invalid='ignore'):
        qP_mmol = np.where(X_tot > 0, dmmol_P_dt / X_tot, 0)

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(t, qP_mmol, label='SP (mmol/g·h)', color='tab:blue')
    plt.xlabel('Time (h)')
    plt.ylabel('Specific Productivity (mmol D-DIBOA / gCDW·h)')
    plt.title('Instantaneous Specific Productivity Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Peak specific productivity: {np.nanmax(qP_mmol):.3f} mmol/g·h")
































def plot_rates(model, t, y):
    S, X, A, O, Pr, E, I = y.T[0], y.T[1], y.T[3], y.T[4], y.T[5], y.T[6], y.T[7]

    def setup_plot(title, y_label, color, y_values):
        plt.figure(figsize=(8, 5))
        plt.plot(t, y_values, label=title, color=color, linewidth=2)
        plt.xlabel("Time (h)", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ks: Substrate uptake rate [1/h]
    ks_vals = []
    for i in range(len(t)):
        s, x, a, o, pr, e = S[i], X[i], A[i], O[i], Pr[i], E[i]
        if x <= 0 or s <= 0:
            ks_vals.append(0)
            continue
        ks = (model.ksmax * s / (model.Ks + s + s**2 / model.Kis)) * (o / (o + model.KoHs)) / \
             ((1 + a / model.Kia)*(1 + pr / model.KPrI)*(1 + e / model.KEI))
        ks_vals.append(max(0, ks))
    setup_plot("Substrate Uptake Rate (ks)", "Rate [1/h]", 'tab:blue', ks_vals)

    # kau: Acetate uptake rate [1/h]
    kau_vals = []
    for i in range(len(t)):
        s, a, o = S[i], A[i], O[i]
        ks = (model.ksmax * s / (model.Ks + s + s**2 / model.Kis)) * (o / (o + model.KoHs))
        kau = model.kamax * a / ((a + model.Ka) * (1 + ks / model.Kis)) * (o / (o + model.KoHs))
        kau_vals.append(max(0, kau))
    setup_plot("Acetate Uptake Rate (kau)", "Rate [1/h]", 'tab:orange', kau_vals)

    # kF: Overflow metabolism rate [1/h]
    kF_vals = []
    for i in range(len(t)):
        s, a, o, pr, e = S[i], A[i], O[i], Pr[i], E[i]
        ks = (model.ksmax * s / (model.Ks + s + s**2 / model.Kis)) * (o / (o + model.KoHs)) / \
             ((1 + a / model.Kia)*(1 + pr / model.KPrI)*(1 + e / model.KEI))
        kF = (model.Pamax * ks) / (ks + model.Kap)
        kF_vals.append(max(0, kF))
    setup_plot("Overflow Flux (kF)", "Rate [1/h]", 'tab:red', kF_vals)

    # ko: Oxygen consumption rate [1/h]
    ko_vals = []
    for i in range(len(t)):
        s, a, o, pr, e = S[i], A[i], O[i], Pr[i], E[i]
        ks = (model.ksmax * s / (model.Ks + s + s**2 / model.Kis)) * (o / (o + model.KoHs)) / \
             ((1 + a / model.Kia)*(1 + pr / model.KPrI)*(1 + e / model.KEI))
        kF = (model.Pamax * ks) / (ks + model.Kap)
        ko = (ks - kF) * o / (o + model.Ko)
        ko_vals.append(max(0, ko))
    setup_plot("Oxygen Consumption Rate (ko)", "Rate [1/h]", 'tab:green', ko_vals)

    # rp: Product formation rate [g/L/h]
    rp_vals = []
    for i in range(len(t)):
        x, pr, i_val = X[i], Pr[i], I[i]
        rp = (model.kpc * x * pr / (model.Kpc + pr)) * (model.k1 * i_val / (i_val + model.Kinhibsat) + model.k2)
        rp_vals.append(max(0, rp))
    setup_plot("Product Formation Rate (rp)", "Rate [g/L/h]", 'tab:purple', rp_vals)
    
    
    # μ: Specific growth rate [1/h]
    mu_vals = []
    for i in range(len(t)):
        s, a, o, pr, e = S[i], A[i], O[i], Pr[i], E[i]
        if s <= 0:
            mu_vals.append(0)
            continue
        ks = (model.ksmax * s / (model.Ks + s + s**2 / model.Kis)) * (o / (o + model.KoHs)) / \
             ((1 + a / model.Kia)*(1 + pr / model.KPrI)*(1 + e / model.KEI))
        kF = (model.Pamax * ks) / (ks + model.Kap)
        kau = model.kamax * a / ((a + model.Ka) * (1 + ks / model.Kis)) * (o / (o + model.KoHs))
        inhib = (1 + pr / model.KPrI) * (1 + e / model.KEI) * (1 + a / model.Kia)
        mu = (model.Yxs * (ks - model.maintenance) + model.Yxa * kau + model.Yxf * kF) / inhib
        mu_vals.append(max(0, mu))
    setup_plot("Specific Growth Rate (μ)", "Rate [1/h]", 'tab:brown', mu_vals)
    
def plot_by_percent(model, t, y, MW_P=165.15, MW_X=24.6):
    """
    Plot BY% (Bioproduct Yield percentage) based on biomass over time.
    
    Parameters:
        t: time array
        y: simulation result (shape: n_times × n_states)
        MW_P: molar mass of product (g/mol)
        MW_X: average molar mass of biomass (g/mol), typical range 24-30 g/mol
        model: instance of EcoliFedBatch (for volume)
    """
    if model is None:
        raise ValueError("You must pass the model instance.")

    # Extract values
    P = y[:, 2]  # g/L Product (D-DIBOA)
    X = y[:, 1]  # g/L Biomass
    V = y[:, 8]  # Volume in L

    # Total product and biomass in reactor
    P_total = P * V  # g
    X_total = X * V  # g

    # Convert to moles
    moles_P = P_total / MW_P
    moles_X = X_total / MW_X

    with np.errstate(divide='ignore', invalid='ignore'):
        BY_percent = 100 * (moles_P / moles_X)
        BY_percent = np.nan_to_num(BY_percent, nan=0.0, posinf=0.0, neginf=0.0)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(t, BY_percent, label='BY% (Product / Biomass)', color='tab:blue')
    plt.xlabel('Time (h)')
    plt.ylabel('Bioproduct Yield (%)')
    plt.title('Product Yield (BY%) over Biomass over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()