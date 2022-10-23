# import functions
from maxent_irl import *
from toy_assembly import *

# import python libraries
import pickle
import numpy as np
from copy import deepcopy
import pandas as pd

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
           [0.60, 0.30,	0.60],
           [0.30, 0.60, 0.60],
           [0.80, 0.40, 0.20],
           [0.80, 0.10, 0.50],
           [0.50, 0.80, 0.10],
           [0.50, 0.10, 0.80],
           [0.10, 0.80, 0.50],
           [0.20, 0.40, 0.80],
           [0.60, 0.50, 0.40],
           [0.50, 0.50, 0.50]]

# weights = [[ 0.3,  0.0,  0.0],
#            [ 0.0,  0.3,	 0.0],
#            [ 0.0,  0.0,  0.3],
#            [-0.3,  0.0,  0.0],
#            [ 0.0, -0.3,	 0.0],
#            [ 0.0,  0.0, -0.3],
#            [ 0.2, -0.2,  0.0],
#            [ 0.0,  0.2, -0.2],
#            [-0.2,  0.0,  0.2]]

noisy_weights1 = np.array(weights)
noisy_weights1 += np.random.normal(0., 0.1, noisy_weights1.shape)
# less_noisy_weights[less_noisy_weights < 0.] = 0.
# less_noisy_weights[less_noisy_weights > 1.] = 1.

noisy_weights2 = np.array(weights)
noisy_weights2 += np.random.normal(0, 0.2, noisy_weights2.shape)
# more_noisy_weights[more_noisy_weights < 0.] = 0.
# more_noisy_weights[more_noisy_weights > 1.] = 1.

noisy_weights3 = np.array(weights)
noisy_weights3 += np.random.normal(0, 0.3, noisy_weights3.shape)

noisy_weights4 = np.array(weights)
noisy_weights4 += np.random.normal(0, 0.4, noisy_weights4.shape)

# for i, w in enumerate(noisy_weights4):
#     correct_w = False
#     while not correct_w:
#         new_w = w + np.random.normal(0, 0.4, w.shape)
#         low_w = new_w >= 0.
#         high_w = new_w <= 1.0
#         if low_w.all() and high_w.all():
#             correct_w = True
#             noisy_weights4[i] = new_w

# print(f"Weights: {weights}")
# print(f"Less Noisy Weights: {less_noisy_weights}")
# print(f"More Noisy Weights: {more_noisy_weights}")

canonical_actions = list(range(len(canonical_features)))
complex_actions = list(range(len(complex_features)))

# initialize canonical task
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_actions)
C.enumerate_states()
C.set_terminal_idx()
# all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])

# initialize actual task
X = ComplexTask(complex_features)
X.set_end_state(complex_actions)
X.enumerate_states()
X.set_terminal_idx()
# all_complex_trajectories = X.enumerate_trajectories([complex_actions])

# loop over all users
canonical_demos, complex_demos, noisy_demos1, noisy_demos2, noisy_demos3, noisy_demos4 = [], [], [], [], [], []
for i in range(len(weights)):

    print("=======================")
    print("User:", i)

    # using abstract features
    abstract_features = np.array([C.get_features(state) for state in C.states])
    canonical_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

    canonical_rewards = canonical_abstract_features.dot(weights[i])
    complex_rewards = complex_abstract_features.dot(weights[i])
    noisy_rewards1 = complex_abstract_features.dot(noisy_weights1[i])
    noisy_rewards2 = complex_abstract_features.dot(noisy_weights2[i])
    noisy_rewards3 = complex_abstract_features.dot(noisy_weights3[i])
    noisy_rewards4 = complex_abstract_features.dot(noisy_weights4[i])

    qf_canonical, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
    qf_complex, _, _ = value_iteration(X.states, X.actions, X.transition, complex_rewards, X.terminal_idx)
    qf_noisy1, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards1, X.terminal_idx)
    qf_noisy2, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards2, X.terminal_idx)
    qf_noisy3, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards3, X.terminal_idx)
    qf_noisy4, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards4, X.terminal_idx)

    canonical_demo = rollout_trajectory(C, qf_canonical, canonical_actions)
    complex_demo = rollout_trajectory(X, qf_complex, complex_actions)
    noisy_demo1 = rollout_trajectory(X, qf_noisy1, complex_actions)
    noisy_demo2 = rollout_trajectory(X, qf_noisy2, complex_actions)
    noisy_demo3 = rollout_trajectory(X, qf_noisy3, complex_actions)
    noisy_demo4 = rollout_trajectory(X, qf_noisy4, complex_actions)

    canonical_demos.append(canonical_demo)
    complex_demos.append(complex_demo)
    noisy_demos1.append(noisy_demo1)
    noisy_demos2.append(noisy_demo2)
    noisy_demos3.append(noisy_demo3)
    noisy_demos4.append(noisy_demo4)

    print("Canonical demo:", canonical_demo)
    print("  Complex demo:", complex_demo)
    print("   Noisy demo1:", noisy_demo1)
    print("   Noisy demo2:", noisy_demo2)
    print("   Noisy demo3:", noisy_demo3)
    print("   Noisy demo4:", noisy_demo4)

np.savetxt("data/user_demos/weights.csv", weights)
np.savetxt("data/user_demos/noisy_weights1.csv", noisy_weights1)
np.savetxt("data/user_demos/noisy_weights2.csv", noisy_weights2)
np.savetxt("data/user_demos/noisy_weights3.csv", noisy_weights3)
np.savetxt("data/user_demos/noisy_weights4.csv", noisy_weights4)
np.savetxt("data/user_demos/canonical_demos.csv", canonical_demos)
np.savetxt("data/user_demos/complex_demos.csv", complex_demos)
np.savetxt("data/user_demos/noisy_demos1.csv", noisy_demos1)
np.savetxt("data/user_demos/noisy_demos2.csv",  noisy_demos2)
np.savetxt("data/user_demos/noisy_demos3.csv",  noisy_demos3)
np.savetxt("data/user_demos/noisy_demos4.csv",  noisy_demos4)
# pickle.dump(all_canonical_trajectories, open("data/user_demos/canonical_trajectories.csv", "wb"))
# pickle.dump(all_complex_trajectories, open("data/user_demos/complex_trajectories.csv", "wb"))

print("Done.")
