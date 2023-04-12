#%%
import numpy as np

def check_criteria(objectives, crit_objs, crit_vals):
    """
    Determines if an objective meets a given set of criteria for a set of SOWs

    inputs:
        objectives: np array of all objectives across a set of SOWs
        crit_objs: the column index of the objective in question
        crit_vals: an array containing [min, max] of the values 
    
    returns:
        meets_criteria: an numpy array containing the SOWs that meet both min and max criteria

    """
    
    # check max and min criteria for each objective
    meet_low = objectives[:, crit_objs] >= crit_vals[0]
    meet_high = objectives[:, crit_objs] <= crit_vals[1]

    # check if max and min criteria are met at the same time
    meets_criteria = np.hstack((meet_low, meet_high)).all(axis=1)

    return meets_criteria