# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from assembly_tasks import *
from import_qualtrics import get_qualtrics_survey
from visualize import *

# import python libraries
import os
import json
import pickle
import numpy as np
import pandas as pd
from os.path import exists
from scipy import stats


# ----------------------------------------------- Utility functions ------------------------------------------------- #

# pre-process feature value
def process_val(x):
    if x == "1 (No effort at all)" or x == "1 (No space at all)":
        x = 1.001
    elif x == "7 (A lot of effort)" or x == "7 (A lot of space)":
        x = 6.999
    else:
        x = float(x)

    return x


# load user ratings
def load_features(data, user_idx, feature_idx, action_idx):
    fea_mat = []
    for j in action_idx:
        fea_vec = []
        for k in feature_idx:
            fea_col = k + str(j)
            fea_val = process_val(data[fea_col][user_idx])
            fea_vec.append(fea_val)
        fea_mat.append(fea_vec)
    return fea_mat


# -------------------------------------------------- Experiment ----------------------------------------------------- #

# select algorithm
run_maxent = True
run_bayes = False
run_random_actions = False
run_random_weights = False
online_learning = True

# algorithm parameters
map_estimate = True
custom_prob = False

# debugging flags
test_canonical = False
test_complex = True

# select samples
n_train_samples = 1
n_test_samples = 2

# -------------------------------------------------- Load data ------------------------------------------------------ #

# download data from qualtrics
learning_survey_id = "SV_8eoX63z06ZhVZRA"
data_path = os.path.dirname(__file__) + "/data/"
get_qualtrics_survey(dir_save_survey=data_path, survey_id=learning_survey_id)

# load user data
demo_path = data_path + "Human-Robot Assembly - Learning.csv"
df = pd.read_csv(demo_path)

save_path = "results/hri_post/"

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

# initialize list of scores
predict_scores, random1_scores, random2_scores, worst_scores = [], [], [], []
learned_weights, updated_weights, updated_rand_weights, true_weights = {}, {}, {}, {}
predicted_sequences, running_accs = [], []

# users to consider for evaluation
users = [7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# users = [31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
pref = [["part"],
        ["space"],
        ["space"],
        [],
        ["space"],
        [],
        [],
        [],
        ["part", "space"],
        ["space"],
        ["part", "space"],
        ["part"],
        ["part", "space"],
        ["part"],
        ["part", "space"],
        ["part", "space"],
        [],
        ["space"],
        ["space", "part"],
        [],
        ["space", "part"],
        ["space", "part"]]

# number of users
n_users = len(users)

# iterate over each user
for ui, user_id in enumerate(users):

    user_id = str(user_id)
    idx = df.index[df['Q1'] == user_id][0]
    print("=======================")
    print("Calculating preference for user:", user_id)

    # user ratings for canonical task features
    canonical_q, complex_q = ["Q6_", "Q7_"], ["Q13_", "Q14_", "Q38_"]
    canonical_feature_values = load_features(df, idx, canonical_q, [2, 4, 6, 3, 5, 7])
    # canonical_feature_values = load_features(df, idx, canonical_q, [1, 3, 5, 2, 4, 6])

    # user ratings for actual task features
    _, n_shared_feature_values = np.shape(canonical_feature_values)
    complex_feature_values = load_features(df, idx, complex_q, [3, 8, 15, 16, 4, 9, 10, 11])
    # complex_feature_values = load_features(df, idx, complex_q, [1, 3, 7, 8, 2, 4, 5, 6])
    shared_feature_values = [val[:n_shared_feature_values] for val in complex_feature_values]

    # load canonical task demonstration
    canonical_survey_actions = [0, 3, 1, 4, 2, 5]
    preferred_order = [df[q][idx] for q in ['Q9_1', 'Q9_2', 'Q9_3', 'Q9_4', 'Q9_5', 'Q9_6']]
    if user_id == "29":
        canonical_demo = [5, 2, 0, 3, 1, 4]
    elif user_id == "16":
        canonical_demo = [5, 2, 0, 1, 4, 3]
    else:
        canonical_demo = [a for _, a in sorted(zip(preferred_order, canonical_survey_actions))]

    # initialize canonical task
    C = CanonicalTask(canonical_feature_values)
    C.set_end_state(canonical_demo)
    C.enumerate_states()
    C.set_terminal_idx()

    # compute features for each state
    canonical_features = np.array([C.get_features(state) for state in C.states])
    canonical_features /= np.linalg.norm(canonical_features, axis=0)
    _, n_shared_features = np.shape(canonical_features)

    # canonical demonstration for training
    canonical_user_demo = [canonical_demo]
    canonical_trajectories = get_trajectories(C.states, canonical_user_demo, C.transition)
    print("Canonical demo:", canonical_user_demo)

    # precompute trajectories for bayesian inference
    if run_bayes:
        if exists("data/hr_demos/canonical_trajectories.csv"):
            all_canonical_trajectories = pickle.load(open("data/hr_demos/canonical_trajectories.csv", "rb"))
        else:
            all_canonical_trajectories = C.enumerate_trajectories([canonical_demo])
            pickle.dump(all_canonical_trajectories, open("data/hr_demos/canonical_trajectories.csv", "wb"))
    else:
        all_canonical_trajectories = []

    # load complex task demonstration
    complex_survey_actions = [0, 4, 4, 2, 2, 4, 4, 2, 2, 1, 5, 3, 6, 6, 7, 6, 6]
    complex_survey_qs = ['Q15_1', 'Q15_2', 'Q15_9', 'Q15_7', 'Q15_12', 'Q15_10', 'Q15_11', 'Q15_13', 'Q15_14',
                         'Q15_3', 'Q15_4', 'Q15_8', 'Q15_5', 'Q15_15', 'Q15_6', 'Q15_16', 'Q15_17']
    preferred_order = [int(df[q][idx]) for q in complex_survey_qs]
    complex_demo = []
    for _, a in sorted(zip(preferred_order, complex_survey_actions)):
        # action_counts = [1, 1, 4, 1, 4, 1, 4, 1]
        complex_demo += [a]  # * action_counts[a]

    # initialize an actual task with shared features
    X = ComplexTask(shared_feature_values)
    X.set_end_state(complex_survey_actions)
    X.enumerate_states()
    X.set_terminal_idx()

    # initialize an actual task with the full set of features
    X_add = ComplexTask(complex_feature_values)
    X_add.set_end_state(complex_survey_actions)
    X_add.enumerate_states()
    X_add.set_terminal_idx()

    # compute feature values for each state in actual task with shared features
    shared_features = np.array([X.get_features(state) for state in X.states])
    shared_features /= np.linalg.norm(shared_features, axis=0)

    # compute feature values for each state in actual task with all features
    complex_features = np.array([X_add.get_features(state) for state in X_add.states])
    complex_features /= np.linalg.norm(complex_features, axis=0)

    # complex demonstrations for testing (ground truth)
    complex_user_demo = [complex_demo]
    complex_trajectories = get_trajectories(X.states, complex_user_demo, X.transition)
    print("Complex demo:", complex_user_demo)

    if run_bayes:
        if exists("data/pilot_demos/complex_trajectories.csv"):
            all_complex_trajectories = pickle.load(open("data/pilot_demos/complex_trajectories.csv", "rb"))
        else:
            all_complex_trajectories = X.enumerate_trajectories([complex_demo])
            pickle.dump(all_complex_trajectories, open("data/pilot_demos/complex_trajectories.csv", "wb"))
    else:
        all_complex_trajectories = []

    # --------------------------------------------------------------------------------------------------------------- #

    # pre-compute likelihood of each trajectory for bayesian inference
    complex_likelihoods = []

    # if custom_prob:
    #     if exists("data/pilot_demos/custom_likelihoods.csv") and custom_prob:
    #         complex_likelihoods = np.loadtxt("data/pilot_demos/custom_likelihoods.csv")
    #         complex_qf = np.loadtxt("data/pilot_demos/complex_q_values.csv")
    #     else:
    #         complex_qf = []
    #         for complex_weights in weight_samples:
    #             save_path = "data/pilot_demos/custom_likelihoods.csv"
    #             r = complex_features.dot(complex_weights)
    #             qf, _, _ = value_iteration(X.states, X.actions, X.transition, r, X.terminal_idx)
    #             likelihood = custom_likelihood(X, all_complex_trajectories, qf)
    #             complex_likelihoods.append(likelihood)
    #             complex_qf.append(qf)
    #         np.savetxt("data/pilot_demos/custom_likelihoods.csv", complex_likelihoods)
    #         np.savetxt("data/pilot_demos/complex_q_values.csv", complex_qf)
    # else:
    #     if exists("data/pilot_demos/complex_likelihoods.csv"):
    #         complex_likelihoods = np.loadtxt("data/user_demos/complex_likelihoods.csv")
    #     else:
    #         for complex_weights in weight_samples:
    #             likelihood, _ = boltzman_likelihood(complex_features, all_complex_trajectories, complex_weights)
    #             complex_likelihoods.append(likelihood)
    #         np.savetxt("data/pilot_demos/complex_likelihoods.csv", complex_likelihoods)

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # select initial distribution of weights
    # init = O.Uniform()
    init = O.Constant(0.5)

    # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=1.2))

    transferred_weights, weights_train_update = [], []

    if run_maxent:
        print("Training using Max-Entropy IRL ...")
        canonical_weights = []
        for _ in range(n_train_samples):
            canonical_update = []
            init_weights = init(n_shared_features)
            canonical_update.append(deepcopy(init_weights))
            _, canonical_weight = maxent_irl(C, canonical_features, canonical_trajectories, optim, init_weights)
            transferred_weights.append(canonical_weight)
            canonical_update.append(canonical_weight)
            weights_train_update.append(canonical_update)

        learned_weights[ui] = weights_train_update

    elif run_bayes:
        print("Training using Bayesian IRL ...")

        # sample candidate weights
        if exists("data/hr_demos/weight_samples.csv"):
            weight_samples = np.loadtxt("data/hr_demos/weight_samples.csv")
        else:
            weight_samples = np.random.uniform(0., 1., (n_train_samples, n_shared_features))
            d = 1.  # np.sum(u, axis=1)  # np.sum(u ** 2, axis=1) ** 0.5
            weight_samples = weight_samples / d

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
        # canonical_weights = weight_samples[entropies.index(max_entropy)]
        # all_max_posteriors = [idx for idx, p in enumerate(posteriors) if p == max_posterior]
        # all_max_entropies = [e for idx, e in enumerate(entropies) if idx in all_max_posteriors]
        # max_entropy = max(all_max_entropies)
        # canonical_weights = weight_samples[all_max_posteriors[all_max_entropies.index(max_entropy)]]
        print("Weights have been learned for the canonical task! Hopefully.")

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

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    if run_bayes or run_maxent:
        print("Testing ...")

        if run_bayes:
            weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples, p=posteriors)
            transferred_weights = weight_samples[weight_idx]

        # score for predicting user action at each time step
        predict_score, p_sequences, weights_test_update, running_acc = [], [], [], []
        for transferred_weight in transferred_weights:

            # transfer rewards over shared features
            transfer_rewards_abstract = shared_features.dot(transferred_weight)

            # compute policy for transferred rewards
            qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                                X.terminal_idx)

            if online_learning:
                init = O.Uniform()  # Constant(0.5)
                for n_sample in range(n_test_samples):
                    p_score, p_sequence, _, up_weights, run_acc = online_predict_trajectory(X, complex_user_demo,
                                                                                            transferred_weight,
                                                                                            shared_features,
                                                                                            complex_features,
                                                                                            [],
                                                                                            optim, init,
                                                                                            user_id,
                                                                                            sensitivity=0.0,
                                                                                            consider_options=False)
                    predict_score.append(p_score)
                    running_acc.append(run_acc)
                    p_sequences.append(str(p_sequence))

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

        predict_score = np.mean(predict_score, axis=0)
        predict_scores.append(predict_score)
        predict_sequence = json.loads(max(set(p_sequences), key=p_sequences.count))
        predicted_sequences.append(predict_sequence)
        if online_learning:
            updated_weights[ui] = weights_test_update
            running_acc = np.mean(running_acc, axis=0)
            running_accs.append(running_acc)

        print("\n")
        print("Complex task:")
        print("   demonstration -", complex_user_demo)
        print("     predictions -", predict_sequence)

    # -------------------------------- Training: Learn weights from complex demo ------------------------------------ #

    if test_complex:
        init = O.Constant(0.5)
        complex_weights = []
        for _ in range(n_train_samples):
            init_weights = init(n_shared_features)
            _, complex_weight = maxent_irl(X, shared_features, complex_trajectories, optim, init_weights)
            complex_weights.append(complex_weight)

        true_weights[ui] = complex_weights

        pickle.dump(true_weights, open(save_path + "true_weights.csv", "wb"))

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #

    if run_random_actions:
        # score for randomly selecting an action
        r_score, predict_sequence = random_trajectory(X.states, complex_user_demo, X.transition)
        random1_scores.append(r_score)

    if run_random_weights:
        print("Testing for random weights ...")

        # weight_idx = np.random.choice(range(len(weight_samples)), size=n_test_samples)
        # random_weights = weight_samples[weight_idx]

        if exists(save_path + "learned_weights.csv"):
            canonical_inits = pickle.load(open(save_path + "learned_weights.csv", "rb"))
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
                random_weight = init_prior(n_shared_features)
            random_rewards = shared_features.dot(random_weight)

            if online_learning:
                for _ in range(n_test_samples):
                    r_score, _, _, up_r_weights, run_acc = online_predict_trajectory(X, complex_user_demo,
                                                                                     random_weight,
                                                                                     shared_features,
                                                                                     complex_features,
                                                                                     [],
                                                                                     optim, init_prior,
                                                                                     user_id,
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

        worst_score = min(np.mean(random_score, axis=1))
        worst_scores.append(worst_score)
        random_score = np.mean(random_score, axis=0)
        random2_scores.append(random_score)
        if online_learning:
            updated_rand_weights[ui] = weights_rand_update
            running_acc = np.mean(running_acc, axis=0)
            running_accs.append(running_acc)


# -------------------------------------------------- Save results --------------------------------------------------- #

if run_bayes:
    # np.savetxt(save_path + "weights" + str(n_users) + "_norm_feat_bayes.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_norm_feat_bayes.csv", predict_scores)

if run_maxent:
    pickle.dump(learned_weights, open(save_path + "learned_weights.csv", "wb"))
    pickle.dump(updated_weights, open(save_path + "updated_weights.csv", "wb"))
    np.savetxt(save_path + "predict" + str(n_users) + "_maxent_new_online.csv", predict_scores)
    pickle.dump(predicted_sequences, open(save_path + "sequence" + str(n_users) + "_maxent_new_online.csv", "wb"))

if run_random_actions:
    np.savetxt(save_path + "random" + str(n_users) + "_actions.csv", random1_scores)

if run_random_weights:
    pickle.dump(updated_rand_weights, open(save_path + "updated_rand_weights.csv", "wb"))
    np.savetxt(save_path + "random" + str(n_users) + "_weights_new_online.csv", random2_scores)
    np.savetxt(save_path + "worst" + str(n_users) + "_online_rand_add.csv", worst_scores)

print("Done.")
