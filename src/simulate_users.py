# import functions
from maxent_irl import *
from toy_assembly import *

# import python libraries
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy


# hard-coded feature values of task actions generated my sampling Halton sequence

canonical_features = [[0.837, 0.244, 0.282],
                      [0.212, 0.578, 0.018],
                      [0.712, 0.911, 0.418],
                      [0.462, 0.195, 0.882],
                      [0.962, 0.528, 0.618],
                      [0.056, 0.861, 0.218]]

complex_features = [[0.950, 0.033, 0.180],
                    [0.044, 0.367, 0.900],
                    [0.544, 0.700, 0.380],
                    [0.294, 0.145, 0.580],
                    [0.794, 0.478, 0.780],
                    [0.169, 0.811, 0.041],
                    [0.669, 0.256, 0.980],
                    [0.419, 0.589, 0.241],
                    [0.919, 0.922, 0.441],
                    [0.106, 0.095, 0.641]]

long_features = [[0.770, 0.723, 0.573],
                 [0.145, 0.612, 0.973],
                 [0.645, 0.279, 0.773],
                 [0.395, 0.945, 0.293],
                 [0.895, 0.476, 0.093],
                 [0.083, 0.143, 0.493],
                 [0.583, 0.810, 0.893],
                 [0.333, 0.365, 0.693],
                 [0.833, 0.032, 0.333],
                 [0.208, 0.698, 0.133],
                 [0.708, 0.587, 0.533],
                 [0.458, 0.254, 0.933],
                 [0.958, 0.921, 0.733],
                 [0.051, 0.538, 0.221]]
                 # [0.551, 0.205, 0.021],
                 # [0.301, 0.871, 0.421],
                 # [0.801, 0.427, 0.821],
                 # [0.176, 0.094, 0.621],
                 # [0.676, 0.760, 0.261],
                 # [0.426, 0.649, 0.061],
                 # [0.926, 0.316, 0.461],
                 # [0.114, 0.982, 0.861],
                 # [0.614, 0.513, 0.661],
                 # [0.364, 0.180, 0.381]]

# weights = [[0.60, 0.20, 0.20],
#            [0.20, 0.60, 0.20],
#            [0.20, 0.20, 0.60],
#            [0.80, 0.10, 0.10],
#            [0.10, 0.80, 0.10],
#            [0.10, 0.10, 0.80],
#            [0.40, 0.40, 0.20],
#            [0.40, 0.20, 0.40],
#            [0.20, 0.40, 0.40],
#            [0.40, 0.30, 0.30],
#            [0.30, 0.40, 0.30],
#            [0.30, 0.30, 0.40],
#            [0.60, 0.30, 0.10],
#            [0.60, 0.10, 0.30],
#            [0.30, 0.60, 0.10],
#            [0.10, 0.60, 0.30],
#            [0.30, 0.10, 0.60],
#            [0.10, 0.30, 0.60],
#            [1.00, 0.00, 0.00],
#            [0.00, 0.00, 1.00]]

weights = [[0.80, 0.10, 0.10],
           [0.10, 0.80,	0.10],
           [0.10, 0.10, 0.80],
           [0.60, 0.30, 0.30],
           [0.30, 0.60, 0.30],
           [0.30, 0.30, 0.60],
           [0.80, 0.80, 0.10],
           [0.80, 0.10, 0.80],
           [0.10, 0.80, 0.80],
           [0.60, 0.60, 0.30],
           [0.60, 0.30, 0.60],
           [0.30, 0.60, 0.60],
           [0.80, 0.40, 0.20],
           [0.80, 0.10, 0.50],
           [0.50, 0.80, 0.10],
           [0.50, 0.10, 0.80],
           [0.10, 0.80, 0.50],
           [0.20, 0.40, 0.80],
           [0.60, 0.50, 0.40],
           [0.50, 0.50, 0.50]]

# ---------------------------------------------------- MAIN --------------------------------------------------------- #

# experiment choice
noisy_users = True
anti_users = False
long_task = True

# modify weights
new_weights = []
if noisy_users:
    noise = [0.1, 0.2, 0.3, 0.4]
    for sd in noise:
        noisy_weights = np.array(weights)
        noisy_weights += np.random.normal(0., sd, noisy_weights.shape)
        new_weights.append(noisy_weights)
elif anti_users:
    anti_weights = 1 - np.array(weights)
    new_weights.append(anti_weights)

# initialize canonical task
canonical_actions = list(range(len(canonical_features)))
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_actions)
C.enumerate_states()
C.set_terminal_idx()
# all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])

# initialize actual task
if long_task:
    complex_actions = list(range(len(long_features)))
    X = LongTask(long_features)
else:
    complex_actions = list(range(len(complex_features)))
    X = ComplexTask(complex_features)
X.set_end_state(complex_actions)
X.enumerate_states()
X.set_terminal_idx()
# all_complex_trajectories = X.enumerate_trajectories([complex_actions])

# loop over all users
canonical_demos, complex_demos, new_demos = [], [], []
for i in range(len(weights)):

    print("=======================")
    print("User:", i)

    # using abstract features
    canonical_abstract_features = np.array([C.get_features(state) for state in C.states])
    canonical_abstract_features /= np.linalg.norm(canonical_abstract_features, axis=0)
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

    # compute user demonstrations
    canonical_rewards = canonical_abstract_features.dot(weights[i])
    complex_rewards = complex_abstract_features.dot(weights[i])
    qf_canonical, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
    qf_complex, _, _ = value_iteration(X.states, X.actions, X.transition, complex_rewards, X.terminal_idx)
    canonical_demo = rollout_trajectory(qf_canonical, C.states, C.transition, canonical_actions)
    complex_demo = rollout_trajectory(qf_complex, X.states, X.transition, complex_actions)

    print("Canonical demo:", canonical_demo)
    canonical_demos.append(canonical_demo)

    print("  Complex demo:", complex_demo)
    complex_demos.append(complex_demo)

    # compute noisy demonstrations
    new_user_demos = []
    for j, new_w in enumerate(new_weights):
        new_rewards = complex_abstract_features.dot(new_w[i])
        new_qf, _, _ = value_iteration(X.states, X.actions, X.transition, new_rewards, X.terminal_idx)
        new_demo = rollout_trajectory(new_qf, X.states, X.transition, complex_actions)
        print("   Noisy demo" + str(j+1) + ":", new_demo)
        new_user_demos.append(new_demo)
    new_demos.append(new_user_demos)


save_path = "data/user_demos/"
np.savetxt(save_path + "weights.csv", weights)
np.savetxt(save_path + "canonical_demos.csv", canonical_demos)
np.savetxt(save_path + "complex_demos.csv", complex_demos)

n_users, n_cases, n_actions = np.shape(new_demos)
new_demos = list(np.reshape(new_demos, (n_cases, n_users, n_actions)))
for j, case_demos in enumerate(new_demos):
    np.savetxt(save_path + "new_weights" + str(j+1) + ".csv", new_weights[j])
    np.savetxt(save_path + "new_demos" + str(j+1) + ".csv", case_demos)

print("Done.")
