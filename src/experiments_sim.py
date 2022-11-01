# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from toy_assembly import *

# import python libraries
import os
import pickle
import numpy as np
import pandas as pd
from os.path import exists


# ------------------------------------------------ Feature values --------------------------------------------------- #

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

long_features = [[0.056, 0.861, 0.218],
                 [0.950, 0.033, 0.180],
                 [0.044, 0.367, 0.900],
                 [0.837, 0.244, 0.282],
                 [0.544, 0.700, 0.380],
                 [0.294, 0.145, 0.580],
                 [0.212, 0.578, 0.018],
                 [0.794, 0.478, 0.780],
                 [0.169, 0.811, 0.041],
                 [0.712, 0.911, 0.418],
                 [0.669, 0.256, 0.980],
                 [0.419, 0.589, 0.241],
                 [0.462, 0.195, 0.882],
                 [0.919, 0.922, 0.441],
                 [0.106, 0.095, 0.641],
                 [0.962, 0.528, 0.618]]

_, n_features = np.shape(complex_features)

adverse_features = [[0.950, 0.033, 0.180, 0.777],
                    [0.044, 0.367, 0.900, 0.152],
                    [0.544, 0.700, 0.380, 0.652],
                    [0.294, 0.145, 0.580, 0.402],
                    [0.794, 0.478, 0.780, 0.902],
                    [0.169, 0.811, 0.041, 0.090],
                    [0.669, 0.256, 0.980, 0.590],
                    [0.419, 0.589, 0.241, 0.340],
                    [0.919, 0.922, 0.441, 0.840],
                    [0.106, 0.095, 0.641, 0.215]]

weights_adverse = [[0.60, 0.20, 0.20, 0.60],
                   [0.80, 0.10, 0.10, 0.01],
                   [0.20, 0.60, 0.20, 0.60],
                   [0.10, 0.80, 0.10, 0.01],
                   [0.20, 0.20, 0.60, 0.60],
                   [0.10, 0.10, 0.80, 0.01],
                   [0.40, 0.40, 0.20, 0.20],
                   [0.40, 0.20, 0.40, 0.20],
                   [0.20, 0.40, 0.40, 0.20],
                   [0.40, 0.30, 0.30, 0.40],
                   [0.30, 0.40, 0.30, 0.40],
                   [0.50, 0.30, 0.20, 0.00],
                   [0.50, 0.20, 0.30, 0.00],
                   [0.30, 0.50, 0.20, 0.80],
                   [0.20, 0.50, 0.30, 0.80],
                   [0.30, 0.20, 0.50, 0.15],
                   [0.20, 0.30, 0.50, 0.80]]

# -------------------------------------------------- Experiment ----------------------------------------------------- #

# select algorithm
run_maxent = True
run_bayes = False
run_random_actions = False
run_random_weights = False
online_learning = True

# algorithm parameters
noisy_users = False
map_estimate = True
custom_prob = False

# debugging flags
test_canonical = False
test_complex = False

# select samples
if online_learning:
    n_train_samples = 5  # 15
else:
    n_train_samples = 5  # 30
n_test_samples = 2

# select initial distribution of weights
if exists("data/user_demos/weight_samples.csv"):
    weight_samples = np.loadtxt("data/user_demos/weight_samples.csv")
else:
    weight_samples = np.random.uniform(0., 1., (n_train_samples, n_features))
    d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
    weight_samples = weight_samples / d

# -------------------------------------------------- Load data ------------------------------------------------------ #

# paths
canonical_path = "data/user_demos/canonical_demos.csv"
if noisy_users:
    complex_path = "data/user_demos/noisy_demos1.csv"
else:
    complex_path = "data/user_demos/complex_demos.csv"

# user demonstrations
canonical_demos = np.loadtxt(canonical_path).astype(int)
complex_demos = np.loadtxt(complex_path).astype(int)

n_users, _ = np.shape(canonical_demos)

# ------------------------------------------------------------------------------------------------------------------- #

# initialize list of scores
predict_scores, random_scores = [], []
weights, decision_pts = [], []

# assembly task actions
canonical_actions = list(range(len(canonical_features)))
complex_actions = list(range(len(complex_features)))

# initialize canonical task
C = CanonicalTask(canonical_features)
C.set_end_state(canonical_actions)
C.enumerate_states()
C.set_terminal_idx()

# compute features for each state
canonical_features = np.array([C.get_features(state) for state in C.states])
canonical_features /= np.linalg.norm(canonical_features, axis=0)

# precompute trajectories for bayesian inference
if run_bayes:
    if exists("data/user_demos/canonical_trajectories.csv"):
        all_canonical_trajectories = pickle.load(open("data/user_demos/canonical_trajectories.csv", "rb"))
    else:
        all_canonical_trajectories = C.enumerate_trajectories([canonical_actions])
        pickle.dump(all_canonical_trajectories, open("data/user_demos/canonical_trajectories.csv", "wb"))
else:
    all_canonical_trajectories = []

# initialize an actual task with shared features
X = ComplexTask(complex_features)
X.set_end_state(complex_actions)
X.enumerate_states()
X.set_terminal_idx()

# compute feature values for each state in actual task with shared features
shared_features = np.array([X.get_features(state) for state in X.states])
shared_features /= np.linalg.norm(shared_features, axis=0)

# initialize an actual task with the full set of features
X_add = ComplexTask(adverse_features)
X_add.set_end_state(complex_actions)
X_add.enumerate_states()
X_add.set_terminal_idx()

# compute feature values for each state in actual task with all features
complex_features_add = np.array([X_add.get_features(state) for state in X_add.states])
complex_features_add /= np.linalg.norm(complex_features_add, axis=0)

if run_bayes:
    if exists("data/user_demos/complex_trajectories.csv"):
        all_complex_trajectories = pickle.load(open("data/user_demos/complex_trajectories.csv", "rb"))
    else:
        all_complex_trajectories = X.enumerate_trajectories([complex_actions])
        pickle.dump(all_complex_trajectories, open("data/user_demos/complex_trajectories.csv", "wb"))
else:
    all_complex_trajectories = []

# ------------------------------------------------------------------------------------------------------------------- #

# pre-compute likelihood of each trajectory for bayesian inference
complex_likelihoods = []

# if custom_prob:
#     if exists("data/user_demos/custom_likelihoods.csv") and custom_prob:
#         complex_likelihoods = np.loadtxt("data/user_demos/custom_likelihoods.csv")
#         complex_qf = np.loadtxt("data/user_demos/complex_q_values.csv")
#     else:
#         complex_qf = []
#         for complex_weights in weight_samples:
#             save_path = "data/user_demos/custom_likelihoods.csv"
#             r = complex_features.dot(complex_weights)
#             qf, _, _ = value_iteration(X.states, X.actions, X.transition, r, X.terminal_idx)
#             likelihood = custom_likelihood(X, all_complex_trajectories, qf)
#             complex_likelihoods.append(likelihood)
#             complex_qf.append(qf)
#         np.savetxt("data/user_demos/custom_likelihoods.csv", complex_likelihoods)
#         np.savetxt("data/user_demos/complex_q_values.csv", complex_qf)
# else:
#     if exists("data/user_demos/complex_likelihoods.csv"):
#         complex_likelihoods = np.loadtxt("data/user_demos/complex_likelihoods.csv")
#     else:
#         for complex_weights in weight_samples:
#             likelihood, _ = boltzman_likelihood(complex_features, all_complex_trajectories, complex_weights)
#             complex_likelihoods.append(likelihood)
#         np.savetxt("data/user_demos/complex_likelihoods.csv", complex_likelihoods)

# ------------------------------------------------------------------------------------------------------------------- #

# loop over all users
for ui in range(len(canonical_demos)):

    print("=======================")
    print("User:", ui)

    # canonical demonstrations
    canonical_user_demo = [list(canonical_demos[ui])]
    canonical_trajectories = get_trajectories(C, canonical_user_demo)

    # complex demonstrations (ground truth)
    complex_user_demo = [list(complex_demos[ui])]
    complex_trajectories = get_trajectories(X, complex_user_demo)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))
    init = O.Uniform()

    if run_maxent:
        print("Training using Max-Entropy IRL ...")
        canonical_weights = []
        for _ in range(n_train_samples):
            init_weights = init(n_features)
            _, canonical_weight = maxent_irl(C, canonical_features, canonical_trajectories, optim, init_weights)
            canonical_weights.append(canonical_weight)

    elif run_bayes:
        print("Training using Bayesian IRL ...")
        posteriors, entropies = [], []
        weight_priors = np.ones(len(weight_samples)) / len(weight_samples)
        for n_sample in range(len(weight_samples)):
            sample = weight_samples[n_sample]
            if custom_prob:
                r = canonical_features.dot(sample)
                qf, _, _ = value_iteration(C.states, C.actions, C.transition, r, C.terminal_idx)
                likelihood_all_traj = custom_likelihood(C, all_canonical_trajectories, qf)
                likelihood_user_demo = custom_likelihood(C, canonical_trajectories, qf)
            else:
                likelihood_all_traj, _ = boltzman_likelihood(canonical_features, all_canonical_trajectories, sample)
                likelihood_user_demo, _ = boltzman_likelihood(canonical_features, canonical_trajectories, sample)

            likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_traj)
            bayesian_update = likelihood_user_demo * weight_priors[n_sample]

            # p = likelihood_all_trajectories / np.sum(likelihood_all_trajectories)
            # entropy = np.sum(p*np.log(p))

            posteriors.append(np.prod(bayesian_update))
            entropies.append(np.sum(np.log(likelihood_user_demo)))

        posteriors = list(posteriors / np.sum(posteriors))

        # select the MAP (maximum a posteriori) weight estimate
        max_posterior = max(posteriors)
        canonical_weights = weight_samples[posteriors.index(max_posterior)]
        # max_entropy = max(entropies)
        # canonical_weights = weight_samples[entropies.index(max_entropy)]
        # all_max_posteriors = [idx for idx, p in enumerate(posteriors) if p == max_posterior]
        # all_max_entropies = [e for idx, e in enumerate(entropies) if idx in all_max_posteriors]
        # max_entropy = max(all_max_entropies)
        # canonical_weights = weight_samples[all_max_posteriors[all_max_entropies.index(max_entropy)]]

    else:
        print("Did not learn any weights :(")
        canonical_weights = None

    print("Weights -", canonical_weights)
    weights.append(canonical_weights)

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    if test_canonical:
        canonical_rewards = canonical_features.dot(canonical_weights)
        qf_abstract, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
        predict_sequence_canonical, _ = predict_trajectory(qf_abstract, C.states, canonical_user_demo, C.transition)

        print("\n")
        print("Canonical task:")
        print("     demonstration -", canonical_user_demo)
        print("predict (abstract) -", predict_sequence_canonical)

    # ---------------------------------------- Testing: Predict complex --------------------------------------------- #

    if run_bayes or run_maxent:
        print("Testing ...")

        if map_estimate:
            transferred_weights = canonical_weights
        elif run_bayes:
            weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples, p=posteriors)
            transferred_weights = weight_samples[weight_idx]
        else:
            transferred_weights = []

        # score for predicting user action at each time step
        predict_score = []
        for transferred_weight in transferred_weights:

            # transfer rewards to complex task
            transfer_rewards_abstract = shared_features.dot(transferred_weight)

            # compute policy for transferred rewards
            qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition_list, transfer_rewards_abstract,
                                                X.terminal_idx)

            # score for predicting user action at each time step
            if online_learning:
                for n_sample in range(n_test_samples):
                    init = O.Uniform()  # O.Constant(0.5)
                    p_score, predict_sequence, _, _, _ = online_predict_trajectory(X, complex_user_demo,
                                                                                   all_complex_trajectories,
                                                                                   complex_likelihoods,
                                                                                   transferred_weight,
                                                                                   shared_features,
                                                                                   complex_features_add,
                                                                                   [], [],
                                                                                   optim, init,
                                                                                   ui,
                                                                                   sensitivity=0.0,
                                                                                   consider_options=False)
                    predict_score.append(p_score)
            else:
                p_score, predict_sequence, _ = predict_trajectory(X, qf_transfer,
                                                                  complex_user_demo,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)
                predict_score.append(p_score)

        predict_score = np.mean(predict_score, axis=0)
        predict_scores.append(predict_score)

        print("\n")
        print("Complex task:")
        print("   demonstration -", complex_user_demo)
        print("     predictions -", predict_sequence)

    # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

    if test_complex:
        init_weights = init(n_features)
        complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, shared_features,
                                                                        complex_trajectories,
                                                                        optim, init_weights, eps=1e-2)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #
    if run_random_actions:
        # score for randomly selecting an action
        r_score, predict_sequence = random_predict_trajectory(X, complex_user_demo)
        random_scores.append(r_score)

    elif run_random_weights:
        print("Testing for random weights ...")

        # random_priors = 1 - priors
        # random_priors /= np.sum(random_priors)
        # weight_idx = np.random.choice(range(len(samples)), size=n_test_samples, p=random_priors)[0]

        # weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples)
        # random_weights = weight_samples[weight_idx]

        init_prior = O.Uniform()

        random_score = []
        max_likelihood = - np.inf
        for n_sample in range(n_train_samples):
            random_weight = init_prior(n_features)  # random_weights[n_sample]
            random_rewards = shared_features.dot(random_weight)

            if online_learning:
                # init = O.Constant(0.5)
                for _ in range(n_test_samples):
                    r_score, predict_sequence, _, _, _ = online_predict_trajectory(X, complex_user_demo,
                                                                                   all_complex_trajectories,
                                                                                   complex_likelihoods,
                                                                                   random_weight,
                                                                                   shared_features,
                                                                                   complex_features_add,
                                                                                   [], [],
                                                                                   optim, init_prior,
                                                                                   ui,
                                                                                   sensitivity=0.0,
                                                                                   consider_options=False)
                    random_score.append(r_score)
            else:
                qf_random, _, _ = value_iteration(X.states, X.actions, X.transition_list, random_rewards, X.terminal_idx)
                r_score, predict_sequence, _ = predict_trajectory(X, qf_random,
                                                                  complex_user_demo,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)
                random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random_scores.append(random_score)

# -------------------------------------------------- Save results --------------------------------------------------- #

save_path = "results/stochastic/"

if run_bayes:
    np.savetxt(save_path + "weights" + str(n_users) + "_norm_feat_bayes_ent.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_norm_feat_bayes_ent.csv", predict_scores)

if run_maxent:
    # np.savetxt(save_path + "weights" + str(n_users) + "_maxent_uni.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_maxent_new_online_stochastic_p0.5.csv", predict_scores)

if run_random_actions:
    np.savetxt(save_path + "random" + str(n_users) + "_actions.csv", random_scores)

if run_random_weights:
    np.savetxt(save_path + "random" + str(n_users) + "_weights_new_online_stochastic_p0.7.csv", random_scores)

print("Done.")
