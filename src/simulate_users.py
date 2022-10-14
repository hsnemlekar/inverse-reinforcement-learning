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


noisy_weights1 = np.array(weights)
noisy_weights1 += np.random.normal(0., 0.1, noisy_weights1.shape)
# less_noisy_weights[less_noisy_weights < 0.] = 0.
# less_noisy_weights[less_noisy_weights > 1.] = 1.

noisy_weights2 = np.array(weights)
noisy_weights2 += np.random.normal(0, 0.2, noisy_weights2.shape)

noisy_weights3 = np.array(weights)
noisy_weights3 += np.random.normal(0, 0.3, noisy_weights3.shape)

noisy_weights4 = np.array(weights)
noisy_weights4 += np.random.normal(0, 0.4, noisy_weights4.shape)

# excluded weight
w_red = 2
reduced_weights = np.delete(weights, w_red, axis=1)

# initialize canonical task
canonical_actions = list(range(len(canonical_features)))
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_actions)
C.enumerate_states()
C.set_terminal_idx()
# all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])

canonical_features_red = np.delete(canonical_features, w_red, axis=1)
C_red = CanonicalTask(canonical_features_red)
C_red.set_end_state(canonical_actions)
C_red.enumerate_states()
C_red.set_terminal_idx()

# initialize actual task
complex_actions = list(range(len(complex_features)))
X = ComplexTask(complex_features)
X.set_end_state(complex_actions)
X.enumerate_states()
X.set_terminal_idx()
# all_complex_trajectories = X.enumerate_trajectories([complex_actions])

complex_features_red = np.delete(complex_features, w_red, axis=1)
X_red = ComplexTask(complex_features_red)
X_red.set_end_state(complex_actions)
X_red.enumerate_states()
X_red.set_terminal_idx()

# loop over all users
canonical_shared_demos, complex_shared_demos = [], []
canonical_demos, complex_demos, noisy_demos1, noisy_demos2, noisy_demos3, noisy_demos4 = [], [], [], [], [], []
for i in range(len(weights)):

    print("=======================")
    print("User:", i)

    # using abstract features
    abstract_features = np.array([C.get_features(state) for state in C.states])
    canonical_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
    complex_abstract_features = np.array([X.get_features(state) for state in X.states])
    complex_abstract_features /= np.linalg.norm(complex_abstract_features, axis=0)

    shared_features = np.array([C_red.get_features(state) for state in C_red.states])
    canonical_shared_features = shared_features / np.linalg.norm(shared_features, axis=0)
    complex_shared_features = np.array([X_red.get_features(state) for state in X_red.states])
    complex_shared_features /= np.linalg.norm(complex_shared_features, axis=0)

    canonical_rewards = canonical_abstract_features.dot(weights[i])
    complex_rewards = complex_abstract_features.dot(weights[i])
    # noisy_rewards1 = complex_abstract_features.dot(noisy_weights1[i])
    # noisy_rewards2 = complex_abstract_features.dot(noisy_weights2[i])
    # noisy_rewards3 = complex_abstract_features.dot(noisy_weights3[i])
    # noisy_rewards4 = complex_abstract_features.dot(noisy_weights4[i])

    canonical_shared_rewards = canonical_shared_features.dot(reduced_weights[i])
    complex_shared_rewards = complex_shared_features.dot(reduced_weights[i])

    qf_canonical, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
    qf_complex, _, _ = value_iteration(X.states, X.actions, X.transition, complex_rewards, X.terminal_idx)
    # qf_noisy1, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards1, X.terminal_idx)
    # qf_noisy2, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards2, X.terminal_idx)
    # qf_noisy3, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards3, X.terminal_idx)
    # qf_noisy4, _, _ = value_iteration(X.states, X.actions, X.transition, noisy_rewards4, X.terminal_idx)

    qf_shared_canonical, _, _ = value_iteration(C_red.states, C_red.actions, C_red.transition, canonical_shared_rewards,
                                                C_red.terminal_idx)
    qf_shared_complex, _, _ = value_iteration(X_red.states, X_red.actions, X_red.transition, complex_shared_rewards,
                                              X_red.terminal_idx)

    canonical_demo = rollout_trajectory(qf_canonical, C.states, C.transition, canonical_actions)
    complex_demo = rollout_trajectory(qf_complex, X.states, X.transition, complex_actions)
    # noisy_demo1 = rollout_trajectory(qf_noisy1, X.states, X.transition, complex_actions)
    # noisy_demo2 = rollout_trajectory(qf_noisy2, X.states, X.transition, complex_actions)
    # noisy_demo3 = rollout_trajectory(qf_noisy3, X.states, X.transition, complex_actions)
    # noisy_demo4 = rollout_trajectory(qf_noisy4, X.states, X.transition, complex_actions)

    canonical_shared_demo = rollout_trajectory(qf_shared_canonical, C_red.states, C_red.transition, canonical_actions)
    complex_shared_demo = rollout_trajectory(qf_shared_complex, X_red.states, X_red.transition, complex_actions)

    canonical_demos.append(canonical_demo)
    complex_demos.append(complex_demo)
    # noisy_demos1.append(noisy_demo1)
    # noisy_demos2.append(noisy_demo2)
    # noisy_demos3.append(noisy_demo3)
    # noisy_demos4.append(noisy_demo4)

    canonical_shared_demos.append(canonical_shared_demo)
    complex_shared_demos.append(complex_shared_demo)

    print("Canonical demo:", canonical_demo)
    print("  Complex demo:", complex_demo)
    # print("   Noisy demo1:", noisy_demo1)
    # print("   Noisy demo2:", noisy_demo2)
    # print("   Noisy demo3:", noisy_demo3)
    # print("   Noisy demo4:", noisy_demo4)

    print(" C Shared demo:", canonical_shared_demo)
    print(" X Shared demo:", complex_shared_demo)


# np.savetxt("data/user_demos/canonical_shared_demos" + str(w_red) + ".csv", canonical_shared_demos)
# np.savetxt("data/user_demos/complex_shared_demos" + str(w_red) + ".csv", complex_shared_demos)
# np.savetxt("data/user_demos/weights.csv", weights)
# np.savetxt("data/user_demos/noisy_weights1.csv", noisy_weights1)
# np.savetxt("data/user_demos/noisy_weights2.csv", noisy_weights2)
# np.savetxt("data/user_demos/noisy_weights3.csv", noisy_weights3)
# np.savetxt("data/user_demos/noisy_weights4.csv", noisy_weights4)
# np.savetxt("data/user_demos/canonical_demos.csv", canonical_demos)
# np.savetxt("data/user_demos/complex_demos.csv", complex_demos)
# np.savetxt("data/user_demos/noisy_demos1.csv", noisy_demos1)
# np.savetxt("data/user_demos/noisy_demos2.csv",  noisy_demos2)
# np.savetxt("data/user_demos/noisy_demos3.csv",  noisy_demos3)
# np.savetxt("data/user_demos/noisy_demos4.csv",  noisy_demos4)
# pickle.dump(all_canonical_trajectories, open("data/user_demos/canonical_trajectories.csv", "wb"))
# pickle.dump(all_complex_trajectories, open("data/user_demos/complex_trajectories.csv", "wb"))

print("Done.")
