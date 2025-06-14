import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp



class EcoliFedBatch:
    def __init__(self):
       # 1. Growth and Yield Parameters
       self.Yxs = 0.55     # g/g Biomass yield from substrate
       self.Yaf = 0.4      # g/g Acetate yield from substrate
       self.Yxa = 0.55      # g/g Biomass yield from acetate
       self.Yoa = 1.08      # g/g Oxygen yield from acetate uptake
       self.Yxf = 0.3      # g/g Biomass yield from overflow 

         
       # 2. Kinetic Rate Constants
       self.ksmax = 2      # h⁻¹ (substrate uptake max)  
       self.kamax = 0.1      # h⁻¹ (acetate uptake is slower)  
       self.Pamax = 0.22     # h⁻¹ (overflow onset)  
       self.kpc = 0.05     # h⁻¹ (whole‐cell catalyst rate)  
       self.kd   = 0.0001     # h⁻¹ (cell‐death rate)  
       self.maintenance = 0.01  # h⁻¹ (maintenace)

        # 2b. Inducer kinetics
       self.k1 = 1.0         # h⁻¹ (inducer‐dependent enhancement)  
       self.k2 = 0.01       # h⁻¹ (basal activity)
       
        
        # 3. Inhibition and Saturation Constants
       self.Ks = 0.03      # g/L Substrate half-saturation constant
       self.Kis = 19.95       # g/L Substrate inhibition constant (high: avoid inhibition until very high S)
       self.Kia = 9      # g/L Acetate inhibition constant
       self.Ka = 0.01       # g/L Acetate half-saturation constant
       self.Kap = 0.5      # g/L Acetate Production half-saturation constant constant
       self.KPrI = 0.9     # g/L Precursor inhibition constant for biomass growth
       self.KEI = 30      # g/L Ethanol inhibition constant
       self.Kpc = 0.01     # g/L catalyst half saturation constant
       self.Ko = 0.0000001      # g/L numerical stability
       self.Kinhibsat = 0.02     # g/L inducer half‐saturation / inhibition constant
       self.KoHs = 0.0005          # g/L oxygen half-saturation constant
       self.KI = 2          #g/L inhibtion by ressources going to the overexpression due to the inducer. .  
       
        # 4. Oxygen Transfer and Reaction Parameters
       self.kla = 200      # 1/h Oxygen transfer coefficient (typical for well-aerated systems)
       self.O_sat = 0.008   # g/L Oxygen saturation concentration in aqueous phase at 37°C & 1 atm air
        
        # 6. Initial Conditions
       self.S0 = 10         # g/L Initial substrate concentration
       self.X0 = 2       # g/L Initial biomass concentration
       self.P0 = 0         # g/L Initial NfsB concentration
       self.A0 = 0         # g/L Initial acetate concentration
       self.O0 = self.O_sat     # g/L Initial oxygen concentration (saturated)
       self.Pr0 = 0        # g/L Initial precursor concentration
       self.E0 = 0         # g/L Initial ethanol concentration
       self.I0 = 0          # g/L initial inducer
       self.V0 = 1.0       # L Initial reactor volume
       self.V_max = 200.0  # L size of reactor
       
       # 7. Feeding Strategy Parameters
       self.F0 =0.021    # L/h Feed flow rate (initial)
       self.Sf = 98.9   # g/L Feed substrate concentration (glucose-rich)
       self.t_start_feed = 1  # h Time to start feeding
       self.k_exp = 0.0001  # 1/h Feed ramp rate ( exponential increase)
       self.If = 100     # g/L inducer in feed
       self.Fi0 = 0.001         # L/h Inducer Feed flow rate (initial)
       
       self.gPgPr = 0.7336 #g/g precursor to product
       self.gEgPr = 0.2046 #g/g precursor to ethanol


       
       # 8. Simulation Time Parameters
       self.t_start = 0 #start of simulation 
       self.t_end =72  #End of process
       self.steps = 20000 #steps of simulation
       self.t0 =22.5 #start time for pulses
       self.dt =6.2 #time between pulses
       self.t_start_inducer_feed = 3 #at what time to add the inducer

       # 9. Precursor batch addition parameters
       
       self.precursor_batch_mass =  9.878   # g per batch
       self.precursor_batch_duration = 6.475        # Duration (h) of the pulse
       self.number_of_batches =8   #Number of pulses
       self.precursor_batch_times = np.array([self.t0 + i * self.dt for i in range(self.number_of_batches)])  # Hours at which to add precursor pulses
       
    def feed_rate(self, t):
        if t < self.t_start_feed:
            return 0

        return self.F0 * np.exp(self.k_exp * (t - self.t_start_feed))
    
    def feed_rate_inducer(self, t):
        if t < self.t_start_inducer_feed: 
            return 0
        return self.Fi0 * np.exp(self.k_exp*(t-self.t_start_inducer_feed))


    def balances(self, t, C):
        S, X, P, A, O, Pr, E, I, V = C
       
        
        Fin = self.feed_rate(t)
        Fiin = self.feed_rate_inducer(t)
        Fout = 0
        F = Fin + Fiin - Fout
  
        if V >= self.V_max:
            Fin = 0
            Fiin = 0
            F = Fin + Fiin - Fout
           
        # Precursor pulse: add precursor in batches at defined times
        # new: add the batch amount only once at the start of each pulse
        
        pulse_mass_rate = 0.0  # g/h
        for tb in self.precursor_batch_times:
            if tb <= t < tb + self.precursor_batch_duration:
                pulse_mass_rate = self.precursor_batch_mass / self.precursor_batch_duration
                break
        #More kinetics
        
        
        # Calculate the specific substrate uptake rate (ks, 1/h)
        ks = max(0, (self.ksmax * S / (self.Ks + S + S**2/self.Kis)) * (O / (O + self.KoHs))/ ((1 + A/self.Kia)*(1 + Pr/self.KPrI)*(1 + E/self.KEI))) 

    
        # Calculate the specific acetate uptake rate (kau, 1/h)
        kau = max(0, self.kamax * A / ((A + self.Ka) * (1 + ks / self.Kis)) * (O / (O + self.KoHs))) 
         
        # Calculate the flux (kF, 1/h) representing the diversion of substrate into an overflow pathway (Acetate)
        kF =(self.Pamax * ks) / (ks + self.Kap)
        
        # Define the oxygen consumption rate (ko, 1/h)
        ko = (ks-kF) * O / (O + self.Ko)
           
        # Calculate the oxygen transfer rate (OTR, g/L·h) from the gas phase to the liquid phase. Could be modelled?
        OTR = self.kla * ( self.O_sat-O)
        
        #rate of production for the product. 
        rp = (self.kpc * X * Pr/(self.Kpc + Pr))* (self.k1 * I/(I + self.Kinhibsat) + self.k2)
        
        #Growth rate
        inhib = (1 + Pr/self.KPrI)*(1 + E/self.KEI)*(1 + A/self.Kia)*(1+I/self.KI)
        mu = max(0,(self.Yxs*(ks - self.maintenance) + self.Yxa*kau + self.Yxf*kF) / inhib)
        
        
        # Mass balances
        # Mass balance for biomass (X)
        # Growth, cell death, and dilution
        dXdt = mu * X - self.kd * X - ((F)/V) * X 
        
        # Mass balance for substrate (S)
        # Substrate change due to feed addition and consumption
        dSdt = ((F)/V)*(self.Sf - S) - (ks + kF) * X
        
        # Mass balance for DD(P)
        # Formation via catalyst reaction and dilution
        dPdt = rp*self.gPgPr - ((F))/V * P
       
        # Mass balance for Acetate (A)
        # Consumption via uptake (kau), dilution, and formation from overflow metabolism (scaled by Yas)        
        dAdt = -kau * X - ((F/V)) * A + kF * X * self.Yaf

        # Mass balance for oxygen (O)
        # Inflow by transfer (OTR) minus consumption by oxidative metabolism (Both substrate and acetate)
        dOdt = OTR - ko * X - self.Yoa * kau * X - ((F) / V) * O
       
        # Mass balance for precursor (Pr)
        # Precursor is consumed in the catalyst reaction (rp), diluted by feed, and increased by precursor pulses
        dPrdt = -rp - ((F)/V) * Pr + pulse_mass_rate/V
      
        # Mass balance for ethanol (E)
        # Formed in parallel with the catalyst reaction (rp) minus dilution
        dEdt = rp*self.gEgPr - ((F)/V) * E
        
        #Mass balance for Inducer (I)
        #added with
        dIdt = (Fiin / V) * self.If - ((F) / V) * I
        
        # Mass balance for reactor volume (V)
        # Increases with the feed rate
        dVdt = F

        return [dSdt, dXdt, dPdt, dAdt, dOdt, dPrdt, dEdt, dIdt, dVdt]
    

    def solve(self):
                # Build the time grid and initial vector
                t_eval = np.linspace(self.t_start, self.t_end, self.steps)
                C0 = [
                    self.S0, self.X0, self.P0, self.A0,
                    self.O0, self.Pr0, self.E0, self.I0, self.V0
                ]
                # Call LSODA via solve_ivp
                sol = solve_ivp(
                    fun=self.balances,
                    t_span=(self.t_start, self.t_end),
                    y0=C0,
                    method='BDF',
                    t_eval=t_eval,
                    rtol=1e-6,
                    atol=1e-8
                )
                # Return (times, solution matrix shaped (n_times, n_states))
                return sol.t, sol.y.T
