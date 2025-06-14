from E_model import EcoliFedBatch

from Optimizing_algorithm_with_concern_to_Pr_and_S_and_t import optimize_Pr_and_S_t_I
from Pareto_opti import run_pareto_optimization, printpareto

from montecarloplots import montecarloplots
from Sensitivty_and_uncertainty import sensitivity_and_uncertainty
from Crazy_long_sensitivity import SALib

from plotfunc import plot,plot_biotransformation_yield, plot_specific_productivity, plot_by_percent, plot_acetate, plot_precursor_DDIBOA, plot_simulation,plot_substrate_and_biomass, plot_biomass_only, plot_product_per_biomass, plot_rates 




import warnings
import time


#Only uncomment what you want to run. THe console and plots gets really crowded otherwise.
start_time = time.time()
warnings.filterwarnings("ignore", message="differential_evolution: the 'workers' keyword has overridden updating='immediate'")

# need a model to take the parameters from 
model = EcoliFedBatch()

# Run the model for the plots
t, C = model.solve()

# Plot all varibels together
#plot(t, C)

# Plot acetate separately
#plot_acetate(t, C)

# Plot precursor and D-DIBOA together
#plot_precursor_DDIBOA(t, C)

#General plots
#plot_simulation(t, C)

#specefic plots
#plot_substrate_and_biomass(t, C)
#plot_biomass_only(t, C)
#plot_rates(model, t, C)
#plot_by_percent(model, t, C)
#plot_biotransformation_yield(model, t, C)
#plot_specific_productivity(model, t, C)


#Sensitivity_and_uncertainty
sensitivity_and_uncertainty(model)
#montecarloplots(model)
#SALib(model)

# Optimzing
if __name__ == "__main__":
    #optimize_Pr_and_S_t_I(model,generations=100,pop_size=40) 
    print("Time for Pareto")
    #run_pareto_optimization(model,generations=100,pop_size=1000,seed=42,plot=True)
    #printpareto(model,1)
    print("Done")
    


end_time = time.time()

tt = (end_time - start_time)/60
