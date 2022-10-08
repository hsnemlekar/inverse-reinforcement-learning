# import functions
import optimizer as O  # stochastic gradient descent optimizer
from maxent_irl import *
from assembly_tasks import *
from import_qualtrics import get_qualtrics_survey

# import python libraries
import os
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
run_random_actions = False
run_random_weights = False
online_learning = False

# algorithm parameters
map_estimate = True

# debugging flags
test_canonical = False
test_complex = False

# select samples
n_train_samples = 1
n_test_samples = 1

# -------------------------------------------------- Load data ------------------------------------------------------ #

# download data from qualtrics
learning_survey_id = "SV_8eoX63z06ZhVZRA"
data_path = os.path.dirname(__file__) + "/data/"
get_qualtrics_survey(dir_save_survey=data_path, survey_id=learning_survey_id)

# load user data
demo_path = data_path + "Human-Robot Assembly - Learning.csv"
df = pd.read_csv(demo_path)

# online_weights = np.loadtxt("results/corl/weights_final10_maxent_online_uni.csv")

# ------------------------------------------- Training: Learn weights ----------------------------------------------- #

# initialize list of scores
predict_scores, random1_scores, random2_scores, worst_scores = [], [], [], []
weights, final_weights = [], []

# users to consider for evaluation
users = [99]
n_users = len(users)
add_pref = [["space"]]

# iterate over each user
for ui, user_id in enumerate(users):

    user_id = str(user_id)
    idx = df.index[df['Q1'] == user_id][0]
    print("=======================")
    print("Calculating preference for user:", user_id)

    # user ratings for canonical task features
    canonical_feature_values = [[2.0, 2.0],  # insert long bolt
                                [2.0, 2.0],  # insert short bolt
                                [4.0, 3.0],  # insert short wire
                                [5.0, 2.0],  # screw long bolt
                                [3.0, 2.0],  # screw short bolt
                                [4.0, 7.0]]  # insert long wire
    # canonical_feature_values = (np.array(canonical_feature_values) - 1.0)/(7.0 - 1.0)

    # user ratings for actual task features
    complex_feature_values = [[4.0, 4.0, 7.0],  # insert main wing
                              [3.0, 5.0, 7.0],  # insert tail wing
                              [2.0, 3.0, 2.0],  # insert long bolt
                              [2.0, 3.0, 2.0],  # insert tail screw
                              [2.0, 4.0, 2.0],  # screw long bolt
                              [3.0, 5.0, 2.0],  # screw tail screw
                              [2.0, 3.0, 2.0],  # screw propeller
                              [2.0, 3.0, 2.0]]  # screw propeller base
    # complex_feature_values = (np.array(complex_feature_values) - 1.0) / (7.0 - 1.0)

    _, n_shared_feature_values = np.shape(canonical_feature_values)
    shared_feature_values = [val[:n_shared_feature_values] for val in complex_feature_values]

    # load canonical task demonstration
    # canonical_demo = [5, 0, 3, 2, 1, 4]
    # canonical_demo = [5, 2, 0, 1, 3, 4]
    canonical_demo = [1, 0, 4, 3, 2, 5]
    #     0 - insert long bolt
    #     1 - insert short bolt
    #     2 - insert wire (short)
    #     3 - screw long bolt
    #     4 - screw short bolt
    #     5 - insert wire (long)

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

    # load complex task demonstration
    # complex_demo = [6, 6, 6, 6, 7, 0, 2, 2, 2, 2, 4, 4, 4, 4, 1, 3, 5]
    complex_demo = [0, 2, 2, 2, 2, 4, 4, 4, 4, 1, 3, 5, 6, 6, 6, 6, 7]
    # complex_demo = [1, 3, 5, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 7]
    #     0 - insert main wing
    #     1 - insert tail wing
    #     2 - insert long bolt into main wing
    #     3 - insert long bolt into tail wing
    #     4 - screw long bolt into main wing
    #     5 - screw long bolt into tail wing
    #     6 - screw propeller
    #     7 - screw propeller base

    # initialize an actual task with shared features
    X = ComplexTask(shared_feature_values)
    X.set_end_state(complex_demo)
    X.enumerate_states()
    X.set_terminal_idx()

    # initialize an actual task with the full set of features
    # TODO: extend code to more than one additional feature
    X_add = ComplexTask(complex_feature_values)
    X_add.set_end_state(complex_demo)
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

    # ---------------------------------------- Training: Learn weights ---------------------------------------------- #

    # select initial distribution of weights
    init = O.Constant(0.5)  # O.Uniform()

    # choose our optimization strategy: exponentiated stochastic gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.6))

    if run_maxent:
        print("Training using Max-Entropy IRL ...")
        init_weights = init(n_shared_features)
        _, canonical_weights = maxent_irl(C, canonical_features, canonical_trajectories, optim, init_weights)
        print("Weights have been learned for the canonical task! Hopefully.")
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

    # ----------------------------------------- Testing: Predict complex -------------------------------------------- #

    if run_maxent:
        print("Testing ...")

        if map_estimate:
            transferred_weights = [canonical_weights]
        else:
            transferred_weights = []

        # score for predicting user action at each time step
        predict_score = []
        for transferred_weight in transferred_weights:

            # transfer rewards over shared features
            transfer_rewards_abstract = shared_features.dot(transferred_weight)

            # compute policy for transferred rewards
            qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, transfer_rewards_abstract,
                                                X.terminal_idx)

            if online_learning:
                init = O.Constant(0.5)
                p_score, predict_sequence, _, _, _ = online_predict_trajectory(X, complex_user_demo,
                                                                               transferred_weight,
                                                                               shared_features,
                                                                               complex_features,
                                                                               add_pref[ui],
                                                                               optim, init,
                                                                               user_id,
                                                                               sensitivity=0.0,
                                                                               consider_options=False)
            else:
                p_score, predict_sequence, _ = predict_trajectory(qf_transfer, X.states,
                                                                  complex_user_demo,
                                                                  X.transition,
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
        init_weights = init(n_shared_features)
        complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, complex_features,
                                                                        complex_trajectories,
                                                                        optim, init_weights, eps=1e-2)

    # ----------------------------------------- Testing: Random baselines ------------------------------------------- #

    if run_random_actions:
        # score for randomly selecting an action
        r_score, predict_sequence = random_trajectory(X.states, complex_user_demo, X.transition)
        random1_scores.append(r_score)

    if run_random_weights:
        print("Testing for random weights ...")

        init_prior = O.Uniform()

        random_score = []
        max_likelihood = - np.inf
        for n_sample in range(n_test_samples):
            random_weight = init_prior(n_shared_features)  # random_weights[n_sample]
            random_rewards = shared_features.dot(random_weight)

            if online_learning:
                r_score, predict_sequence, _, _, _ = online_predict_trajectory(X, complex_user_demo,
                                                                               random_weight,
                                                                               shared_features,
                                                                               complex_features,
                                                                               [],
                                                                               optim, init_prior,
                                                                               user_id,
                                                                               sensitivity=0.0,
                                                                               consider_options=False)
            else:
                qf_random, _, _ = value_iteration(X.states, X.actions, X.transition, random_rewards, X.terminal_idx)
                r_score, predict_sequence, _ = predict_trajectory(qf_random, X.states,
                                                                  complex_user_demo,
                                                                  X.transition,
                                                                  sensitivity=0.0,
                                                                  consider_options=False)

            random_score.append(r_score)

        random_score = np.mean(random_score, axis=0)
        random2_scores.append(random_score)

# -------------------------------------------------- Save results --------------------------------------------------- #

save_path = "results/corl/"

if run_maxent:
    np.savetxt(save_path + "weights" + str(n_users) + "_maxent_new_online.csv", weights)
    np.savetxt(save_path + "predict" + str(n_users) + "_maxent_new_online.csv", predict_scores)

if run_random_actions:
    np.savetxt(save_path + "random" + str(n_users) + "_actions.csv", random1_scores)

if run_random_weights:
    np.savetxt(save_path + "random" + str(n_users) + "_weights_online_rand_add.csv", random2_scores)
    np.savetxt(save_path + "worst" + str(n_users) + "_online_rand_add.csv", worst_scores)

print("Done.")
