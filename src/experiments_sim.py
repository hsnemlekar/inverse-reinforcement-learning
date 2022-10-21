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
from copy import deepcopy


# -------------------------------------------------- Experiment ----------------------------------------------------- #

# select algorithm
run_maxent = False
run_bayes = False
run_random_actions = False
run_random_weights = True
online_learning = True
add_feature = False

# experiment parameters
missing_feature = []
long_task = False
noisy_users = False
map_estimate = True
custom_prob = False

# debugging flags
test_canonical = False
test_complex = False

# select iterations
n_train_samples = 10
n_test_samples = 2

# -------------------------------------------------- Load data ------------------------------------------------------ #

# paths
data_path = "data/user_demos/"
weights_path = data_path + "weights.csv"
canonical_features_path = data_path + "canonical_features.csv"
complex_features_path = data_path + "complex_features.csv"
if add_feature:
    canonical_demo_path = data_path + "canonical_shared_demos" + str(missing_feature) + ".csv"
    # complex_demo_path = data_path + "complex_shared_demos" + str(missing_feature) + ".csv"
    complex_demo_path = data_path + "complex_demos.csv"
elif noisy_users:
    canonical_demo_path = data_path + "canonical_demos.csv"
    complex_demo_path = data_path + "noisy_demos1.csv"
    # complex_demo_path = data_path + "new_demos1.csv"
else:
    canonical_demo_path = data_path + "canonical_demos.csv"
    complex_demo_path = data_path + "complex_demos.csv"

# load feature values and user demonstrations
true_weights = np.loadtxt(weights_path).astype(float)
canonical_features = np.loadtxt(canonical_features_path).astype(float).tolist()
complex_features = np.loadtxt(complex_features_path).astype(float).tolist()
canonical_demos = np.loadtxt(canonical_demo_path).astype(int)
complex_demos = np.loadtxt(complex_demo_path).astype(int)

# select initial distribution of weights (for bayesian learning)
if exists("data/user_demos/weight_samples.csv"):
    weight_samples = np.loadtxt("data/user_demos/weight_samples.csv")
else:
    n_samples = 100
    _, n_features = np.shape(complex_features)
    weight_samples = np.random.uniform(0., 1., (n_samples, n_features))
    d = np.sum(weight_samples, axis=1)  # np.sum(weight_samples ** 2, axis=1) ** 0.5
    weight_samples = weight_samples / d
    np.savetxt("data/user_demos/weight_samples.csv", weight_samples)

# ------------------------------------------------------------------------------------------------------------------- #

# initialize list of scores
predict_scores, random_scores = [], []
learned_weights, updated_weights, updated_rand_weights, running_accs = {}, {}, {}, []

# assembly task actions
canonical_actions = list(range(len(canonical_features)))
complex_actions = list(range(len(complex_features)))

# initialize canonical task
reduced_features = np.delete(canonical_features, missing_feature, axis=1)
C = CanonicalTask(reduced_features)
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
reduced_features = np.delete(complex_features, missing_feature, axis=1)
X = ComplexTask(reduced_features)
X.set_end_state(complex_actions)
X.enumerate_states()
X.set_terminal_idx()

# compute feature values for each state in actual task with shared features
shared_features = np.array([X.get_features(state) for state in X.states])
shared_features /= np.linalg.norm(shared_features, axis=0)

# initialize an actual task with the full set of features
X_add = ComplexTask(complex_features)
X_add.set_end_state(complex_actions)
X_add.enumerate_states()
X_add.set_terminal_idx()

# compute feature values for each state in actual task with all features
complex_features = np.array([X_add.get_features(state) for state in X_add.states])
complex_features /= np.linalg.norm(complex_features, axis=0)

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
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)

    # complex demonstrations (ground truth)
    complex_user_demo = [list(complex_demos[ui])]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    _, n_features = np.shape(canonical_features)

    # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=1.2))
    init = O.Uniform()

    transferred_weights, weights_train_update = [], []

    if run_maxent:
        print("Training using Max-Entropy IRL ...")
        for _ in range(n_train_samples):
            canonical_update = []
            init_weights = init(n_features)
            canonical_update.append(deepcopy(init_weights))
            _, canonical_weight = maxent_irl(C, canonical_features, canonical_trajectories, optim, init_weights)
            transferred_weights.append(canonical_weight)
            canonical_update.append(canonical_weight)
            weights_train_update.append(canonical_update)

        learned_weights[ui] = weights_train_update

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
        transferred_weights = weight_samples[posteriors.index(max_posterior)]
        # max_entropy = max(entropies)
        # transferred_weights = weight_samples[entropies.index(max_entropy)]
        # all_max_posteriors = [idx for idx, p in enumerate(posteriors) if p == max_posterior]
        # all_max_entropies = [e for idx, e in enumerate(entropies) if idx in all_max_posteriors]
        # max_entropy = max(all_max_entropies)
        # transferred_weights = weight_samples[all_max_posteriors[all_max_entropies.index(max_entropy)]]

    else:
        print("Did not learn any weights :(")
        transferred_weights = None

    print("Weights -", transferred_weights)

    # --------------------------------------- Verifying: Reproduce demo --------------------------------------------- #

    if test_canonical:
        weights_idx = np.random.choice(range(len(transferred_weights)))
        canonical_rewards = canonical_features.dot(transferred_weights[weights_idx])
        qf_abstract, _, _ = value_iteration(C.states, C.actions, C.transition, canonical_rewards, C.terminal_idx)
        predict_sequence_canonical, _ = predict_trajectory(qf_abstract, C.states, canonical_user_demo, C.transition)

        print("\n")
        print("Canonical task:")
        print("     demonstration -", canonical_user_demo)
        print("predict (abstract) -", predict_sequence_canonical)

    # ---------------------------------------- Testing: Predict complex --------------------------------------------- #

    if run_bayes or run_maxent:
        print("Testing ...")

        if run_bayes:
            weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples, p=posteriors)
            transferred_weights = weight_samples[weight_idx]

        # score for predicting user action at each time step
        predict_score, weights_test_update, running_acc = [], [], []
        for transferred_weight in transferred_weights:

            # transfer rewards to complex task
            transfer_rewards_abstract = shared_features.dot(transferred_weight)

            # compute policy for transferred rewards
            qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                                X.terminal_idx)

            # score for predicting user action at each time step
            if online_learning:
                for n_sample in range(n_test_samples):
                    init = O.Uniform()  # O.Constant(0.5)
                    p_score, _, _, up_weights, run_acc = online_predict_trajectory(X, complex_user_demo,
                                                                                   transferred_weight,
                                                                                   shared_features,
                                                                                   complex_features,
                                                                                   [],
                                                                                   optim, init,
                                                                                   ui,
                                                                                   sensitivity=0.0,
                                                                                   consider_options=False)
                    predict_score.append(p_score)
                    running_acc.append(run_acc)

                    test_update = [transferred_weight]
                    test_update += up_weights
                    weights_test_update.append(test_update)

            else:
                p_score, predict_sequence, _ = predict_trajectory(qf_transfer, X.states,
                                                                  complex_user_demo,
                                                                  X.transition,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)
                predict_score.append(p_score)

        # accumulate results
        predict_score = np.mean(predict_score, axis=0)
        predict_scores.append(predict_score)
        if online_learning:
            updated_weights[ui] = weights_test_update
            running_acc = np.mean(running_acc, axis=0)
            running_accs.append(running_acc)

        print("\n")
        print("Complex task:")
        print("   demonstration -", complex_user_demo)
        # print("     predictions -", predict_sequence)

    # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

    if test_complex:
        init_weights = init(n_features)
        complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, shared_features,
                                                                        complex_trajectories,
                                                                        optim, init_weights, eps=1e-2)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #
    if run_random_actions:
        # score for randomly selecting an action
        r_score, predict_sequence = random_trajectory(X.states, complex_user_demo, X.transition)
        random_scores.append(r_score)

    elif run_random_weights:
        print("Testing for random weights ...")

        # weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples)
        # random_weights = weight_samples[weight_idx]

        if exists("results/corl_sim/learned_weights.csv"):
            canonical_inits = pickle.load(open("results/corl_sim/learned_weights.csv", "rb"))
            random_priors = np.array(canonical_inits[ui])[:, 0, :]
        else:
            random_priors = []

        init_prior = O.Uniform()

        random_score, weights_rand_update, running_acc = [], [], []
        max_likelihood = - np.inf
        for n_sample in range(n_train_samples):
            if len(random_priors) > 0:
                random_weight = random_priors[n_sample]
            else:
                random_weight = init_prior(n_features)
            random_rewards = shared_features.dot(random_weight)

            if online_learning:
                # init = O.Constant(0.5)
                for _ in range(n_test_samples):
                    r_score, _, _, up_r_weights, run_acc = online_predict_trajectory(X, complex_user_demo,
                                                                                     random_weight,
                                                                                     shared_features,
                                                                                     complex_features,
                                                                                     [],
                                                                                     optim, init_prior,
                                                                                     ui,
                                                                                     sensitivity=0.0,
                                                                                     consider_options=False)
                    random_score.append(r_score)
                    running_acc.append(run_acc)

                    rand_update = [random_weight]
                    rand_update += up_r_weights
                    weights_rand_update.append(rand_update)

            else:
                qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards, X.terminal_idx)
                r_score, predict_sequence, _ = predict_trajectory(qf_random, X.states,
                                                                  complex_user_demo,
                                                                  X.transition,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)
                random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random_scores.append(random_score)
        if online_learning:
            updated_rand_weights[ui] = weights_rand_update
            running_acc = np.mean(running_acc, axis=0)
            running_accs.append(running_acc)

# -------------------------------------------------- Save results --------------------------------------------------- #

save_path = "results/corl_sim/"
n_users, _ = np.shape(canonical_demos)

if run_bayes:
    # np.savetxt(save_path + "weights" + str(n_users) + "_norm_feat_bayes_ent.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_norm_feat_bayes_ent.csv", predict_scores)

if run_maxent:
    pickle.dump(learned_weights, open(save_path + "learned_weights.csv", "wb"))
    pickle.dump(updated_weights, open(save_path + "updated_weights.csv", "wb"))
    # np.savetxt(save_path + "predict" + str(n_users) + "_maxent_uni_worst.csv", predict_scores)

if run_random_actions:
    np.savetxt(save_path + "random" + str(n_users) + "_actions.csv", random_scores)

if run_random_weights:
    pickle.dump(updated_rand_weights, open(save_path + "updated_rand_weights.csv", "wb"))
    # np.savetxt(save_path + "random" + str(n_users) + "_weights_new_online_add1.csv", random_scores)

print("Done.")
