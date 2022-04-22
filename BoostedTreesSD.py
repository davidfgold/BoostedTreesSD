#%%
import numpy as np
from helper_functions import check_criteria
from sklearn.ensemble import GradientBoostingClassifier
from copy import deepcopy
from matplotlib import pyplot as plt
#%%
# load objectives and create input array of boolean values
Reeval_objectives = np.loadtxt('Model_output.csv', skiprows=1, delimiter=',')
REL = check_criteria(Reeval_objectives, [0], [.979, 1])
RF = check_criteria(Reeval_objectives, [1], [0, 0.10])
WCC = check_criteria(Reeval_objectives, [2], [0, 0.10])
ALL = np.vstack((REL, RF, WCC)).all(axis=0)


#%% 
# load DU factors
DU_factors = np.loadtxt('DU_factors.csv', skiprows=1, delimiter=',')
DU_names = ['Watertown Rest. Eff.', 'Dryville Rest. Eff.', 'Fallsland Rest. Eff.',
            'Demand Growth Rate', 'Bond Term', 'Bond Interest',
            'Discount Rate', 'NRR Perm', 'NRR Const', 'CRR L Perm',
            'CRR L Const.',	'CRR H Perm.', 'CRR H Const.', 'WR1 Perm.',
             'WR1 Const.', 'Inflows A', 'Inflows m','Inflows p']
#%%
# set up the trees
gbc = GradientBoostingClassifier(n_estimators=500,
                                 learning_rate=0.1,
                                 max_depth=4)

gbc.fit(DU_factors, ALL)

feature_importances = deepcopy(gbc.feature_importances_)
importances_sorted_idx = np.argsort(feature_importances)
sorted_names = [DU_names[i] for i in importances_sorted_idx]

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.barh(np.arange(len(feature_importances)), feature_importances[importances_sorted_idx])
ax.set_yticks(np.arange(len(feature_importances)))
ax.set_yticklabels(sorted_names)
ax.set_xlim([0,1])
ax.set_xlabel('Feature Importance')
plt.tight_layout()

#%%
selected_factors = DU_factors[:, [3,0]]
gbc_2_factors = GradientBoostingClassifier(n_estimators=200,
                                 learning_rate=0.1,
                                 max_depth=3)
gbc_2_factors.fit(selected_factors, ALL)

 # plot prediction contours
x_data = selected_factors[:,0]
y_data = selected_factors[:,1]

x_min, x_max = (x_data.min(), x_data.max())
y_min, y_max = (y_data.min(), y_data.max())

xx, yy = np.meshgrid(np.arange(x_min, x_max * 1.001, (x_max - x_min) / 100),
                        np.arange(y_min, y_max * 1.001, (y_max - y_min) / 100))
                        
dummy_points = list(zip(xx.ravel(), yy.ravel()))

z = gbc_2_factors.predict_proba(dummy_points)[:, 1]
z[z < 0] = 0.
z = z.reshape(xx.shape)
        
fig = plt.figure(figsize=(10,8))
ax = fig.gca()
ax.contourf(xx, yy, z, [0, 0.5, 1.], cmap='RdBu',
                alpha=.6, vmin=0.0, vmax=1)
ax.scatter(selected_factors[:,0], selected_factors[:,1],\
            c=ALL, cmap='Reds_r', edgecolor='grey', 
            alpha=.6, s= 100, linewidth=.5)
ax.set_xlim([.5, 2])
ax.set_ylim([.9,1.1])
ax.set_xlabel('Demand Growth Multiplier')
ax.set_ylabel('Restriction Eff. Multiplier')

#%%
xlabel = rdm_names[factor1_idx] + \
' (' + str(int(feature_importances[factor1_idx]*100)) + '%)'
ylabel = rdm_names[factor2_idx]  + \
' (' + str(int(feature_importances[factor2_idx]*100)) + '%)'
#ax.set_xlabel(xlabel, fontsize=8)
#ax.set_ylabel(ylabel, fontsize=8)

ax.set_xlim([min(selected_factors[:,0]), max(selected_factors[:,0])])
ax.set_ylim([min(selected_factors[:,1]), max(selected_factors[:,1])])





