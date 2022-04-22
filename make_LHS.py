#%%
import numpy as np
from smt.sampling_methods import LHS

'''
This script will generate 1000 Latin Hypercube Samples (LHS)
of deeply uncertain system parameters for the Sedento Valley
'''


# create an array storing the ranges of deeply uncertain parameters
DU_factor_limits = np.array([
    [0.9, 1.1], # Watertown restriction efficacy 
    [0.9, 1.1], # Dryville restriction efficacy
    [0.9, 1.1], # Fallsland restriction efficacy
    [0.5, 2.0], # Demand growth rate multiplier
    [1.0, 1.2], # Bond term
    [0.6, 1.0], # Bond interest rate
    [0.6, 1.4], # Discount rate
    [0.75, 1.5], # New River Reservoir permitting time
    [1.0, 1.2], # New River Reservoir construction time
    [0.75, 1.5], # College Rock Reservoir (low) permitting time
    [1.0, 1.2], # College Rock Reservoir (low) construction time
    [0.75, 1.5], # College Rock Reservoir (high) permitting time
    [1.0, 1.2], # College Rock Reseroir (high) construction time
    [0.75, 1.5], # Water Reuse permitting time
    [1.0, 1.2], # Water Reuse construction time
    [0.8, 1.2], # Inflow amplitude
    [0.2, 0.5], # Inflow frequency
    [-1.57, 1.57]]) # Inflow phase

# Use the smt package to set up the LHS sampling
sampling = LHS(xlimits=DU_factor_limits)

# We will create 1000 samples
num = 1000

# Create the actual sample
x = sampling(num)

# save to a csv file
np.savetxt('DU_factors.csv', x, delimiter=',')