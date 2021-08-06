from copy import deepcopy
import numpy as np
from itertools import product  # Cartesian product for iterators
import optimizer as O  # stochastic gradient descent optimizer
from vi import value_iteration
from maxent_irl import *
import pandas as pd

# ------------------------------------------------- Optimization ---------------------------------------------------- #

# choose our parameter initialization strategy:
# initialize parameters with constant
init = O.Constant(1.0)

# choose our optimization strategy:
# we select exponentiated stochastic gradient descent with linear learning-rate decay
optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))


# --------------------------------------------- Full Complex Task --------------------------------------------------- #


class FullComplexTask:
    """
    Full airplane assembly task parameters.
    """

    def __init__(self):
        # actions that can be taken in the complex task
        self.actions = [0,  # insert main wing
                        1,  # insert tail wing
                        2,  # insert right wing tip
                        3,  # insert left wing tip
                        4,  # insert long bolt into main wing
                        5,  # insert long bolt into tail wing
                        6,  # screw long bolt into main wing
                        7,  # screw long bolt into tail wing
                        8,  # screw propeller
                        9,  # screw propeller base
                        10]  # screw propeller cap

        # feature values for each action = [physical_effort, mental_effort]
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale
        features = [[3.6, 2.6],  # insert main wing
                    [2.4, 2.2],  # insert tail wing
                    [1.8, 1.6],  # insert right wing tip
                    [1.8, 1.6],  # insert left wing tip
                    [1.6, 2.0],  # insert long bolt into main wing
                    [1.4, 1.4],  # insert long bolt into tail wing
                    [2.8, 1.8],  # screw long bolt into main wing
                    [2.0, 1.8],  # screw long bolt into tail wing
                    [3.8, 2.6],  # screw propeller
                    [2.2, 1.6],  # screw propeller base
                    [2.2, 2.4]]  # screw propeller cap

        self.features = (np.array(features) - self.min_value) / (self.max_value - self.min_value)

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # terminal state of the assembly (each action has been performed)
        self.s_end = [1, 1, 1, 1, 4, 2, 4, 2, 4, 1, 1]

    def transition(self, s_from, a):
        # preconditions
        if a in [0, 1, 2, 3] and s_from[a] < 1:
            p = 1.0
        elif a == 4 and s_from[a] < 4 and s_from[0] == 1:
            p = 1.0
        elif a == 5 and s_from[a] < 2 and s_from[1] == 1:
            p = 1.0
        elif a == 6 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 7 and s_from[a] < 2 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 8 and s_from[a] < 4:
            p = 1.0
        elif a == 9 and s_from[a] < 1 and s_from[a - 1] == 4:
            p = 1.0
        elif a == 10 and s_from[a] < 1 and s_from[a - 1]:
            p = 1.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_to = deepcopy(s_from)
            s_to[a] += 1
            return p, s_to
        else:
            return p, None

    def back_transition(self, s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a == 0 and s_to[4] < 1:
                p = 1.0
            elif a == 1 and s_to[5] < 1:
                p = 1.0
            elif a in [4, 5] and s_to[a] > s_to[a + 2]:
                p = 1.0
            elif a in [8, 9] and s_to[a + 1] < 1:
                p = 1.0
            elif a in [2, 3, 6, 7, 10]:
                p = 1.0
            else:
                p = 0.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_from = deepcopy(s_to)
            s_from[a] -= 1
            return p, s_from
        else:
            return p, None


# ----------------------------------------------- Canonical Task ----------------------------------------------------- #

class CanonicalTask:
    """
    Canonical task parameters.
    """

    def __init__(self, list):
        # actions that can be taken in the complex task
        self.actions = [0,  # insert long bolt
                        1,  # insert short bolt
                        2,  # insert wire
                        3,  # screw long bolt
                        4,  # screw short bolt
                        5]  # screw wire

        # feature values for each action = [physical_effort, mental_effort]
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale
        features_old = [[1.1, 1.1],  # insert long bolt
                    [1.1, 1.1],  # insert short bolt
                    [5.0, 5.0],  # insert wire
                    [6.9, 5.0],  # screw long bolt
                    [2.0, 2.0],  # screw short bolt
                    [2.0, 1.1]]  # screw wire
        features_new = [[int((list[0][0])[0]), int((list[1][0])[0])],
                        [int((list[0][1])[0]), int((list[1][1])[0])],
                        [int((list[0][2])[0]), int((list[1][2])[0])],
                        [int((list[0][3])[0]), int((list[1][3])[0])],
                        [int((list[0][4])[0]), int((list[1][4])[0])],
                        [int((list[0][5])[0]), int((list[1][5])[0])]]
        features_ranking = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        list_to_ignore = [-1, -1, -1, -1, -1, -1]
        index_in_ignore = 0
        for i in range(len(features_new)):
            #input("Hit Enter")
            max_cur = 0
            max_index = 0
            for j in range(len(features_new)):
                if(j in list_to_ignore):
                    continue
                if(max_cur < features_new[j][0]):
                    max_cur = features_new[j][0]
                    max_index = j
            #print(max_index)
            features_ranking[max_index][0] = i+1
            list_to_ignore[index_in_ignore] = max_index
            index_in_ignore += 1
            # print("Features_Ranking")
            # print(features_ranking)
            # print("list_to_ignore")
            # print(list_to_ignore)
        list_to_ignore = [-1, -1, -1, -1, -1, -1]
        index_in_ignore = 0
        for i in range(len(features_new)):
            #input("Hit Enter")
            max_cur = 0
            max_index = 0
            for j in range(len(features_new)):
                if(j in list_to_ignore):
                    continue
                if(max_cur < features_new[j][1]):
                    max_cur = features_new[j][1]
                    max_index = j
            #print(max_index)
            features_ranking[max_index][1] = i+1
            list_to_ignore[index_in_ignore] = max_index
            index_in_ignore += 1
            # print("Features_Ranking")
            # print(features_ranking)
            # print("list_to_ignore")
            # print(list_to_ignore)
        
        print(features_ranking)
        features = features_ranking

        self.features = (np.array(features) - self.min_value) / (self.max_value - self.min_value)

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0, 0, 0, 0, 0, 0]

        # terminal state of the assembly (each action has been performed)
        self.s_end = [1, 1, 1, 1, 1, 1]

    def transition(self, s_from, a):
        # preconditions
        if s_from[a] < 1:
            if a in [0, 1, 2]:
                prob = 1.0
            elif a in [3, 4, 5] and s_from[a - 3] == 1:
                prob = 1.0
            else:
                prob = 0.0
        else:
            prob = 0.0

        # transition to next state
        if prob == 1.0:
            s_to = deepcopy(s_from)
            s_to[a] += 1
            return prob, s_to
        else:
            return prob, None

    def back_transition(self, s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a in [0, 1, 2] and s_to[a + 3] < 1:
                p = 1.0
            elif a in [3, 4, 5]:
                p = 1.0
            else:
                p = 0.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_from = deepcopy(s_to)
            s_from[a] -= 1
            return p, s_from
        else:
            return p, None


# ----------------------------------------------- Complex Task ----------------------------------------------------- #


class ComplexTask:
    """
    Complex task parameters.
    """

    def __init__(self):
        # actions that can be taken in the complex task
        self.actions = [0,  # insert main wing
                        1,  # insert tail wing
                        2,  # insert long bolt into main wing
                        3,  # insert long bolt into tail wing
                        4,  # screw long bolt into main wing
                        5,  # screw long bolt into tail wing
                        6,  # screw propeller
                        7]  # screw propeller base

        # feature values for each action = [physical_effort, mental_effort]
        self.min_value, self.max_value = 1.0, 7.0  # rating are on 1-7 Likert scale
        features = [[3.6, 2.6],  # insert main wing
                    [2.4, 2.2],  # insert tail wing
                    [1.6, 2.0],  # insert long bolt into main wing
                    [1.4, 1.4],  # insert long bolt into tail wing
                    [2.8, 1.8],  # screw long bolt into main wing
                    [2.0, 1.8],  # screw long bolt into tail wing
                    [3.8, 2.6],  # screw propeller
                    [2.2, 1.6]]  # screw propeller base

        self.features = (np.array(features) - self.min_value) / (self.max_value - self.min_value)

        # start state of the assembly task (none of the actions have been performed)
        self.s_start = [0, 0, 0, 0, 0, 0, 0, 0]

        # terminal state of the assembly (each action has been performed)
        self.s_end = [1, 1, 4, 2, 4, 2, 4, 1]

    def transition(self, s_from, a):
        # preconditions
        if a in [0, 1] and s_from[a] < 1:
            p = 1.0
        elif a == 2 and s_from[a] < 4 and s_from[0] == 1:
            p = 1.0
        elif a == 3 and s_from[a] < 2 and s_from[1] == 1:
            p = 1.0
        elif a == 4 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 5 and s_from[a] < 2 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 6 and s_from[a] < 4:
            p = 1.0
        elif a == 7 and s_from[a] < 1 and s_from[a - 1] == 4:
            p = 1.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_to = deepcopy(s_from)
            s_to[a] += 1
            return p, s_to
        else:
            return p, None

    def back_transition(self, s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a == 0 and s_to[2] < 1:
                p = 1.0
            elif a == 1 and s_to[3] < 1:
                p = 1.0
            elif a in [2, 3] and s_to[a] > s_to[a + 2]:
                p = 1.0
            elif a in [6] and s_to[a + 1] < 1:
                p = 1.0
            elif a in [4, 5, 7]:
                p = 1.0
            else:
                p = 0.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_from = deepcopy(s_to)
            s_from[a] -= 1
            return p, s_from
        else:
            return p, None

    def transition(self, s_from, a):
        # preconditions
        if a in [0, 1] and s_from[a] < 1:
            p = 1.0
        elif a == 2 and s_from[a] < 4 and s_from[0] == 1:
            p = 1.0
        elif a == 3 and s_from[a] < 2 and s_from[1] == 1:
            p = 1.0
        elif a == 4 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 5 and s_from[a] < 2 and s_from[a] + 1 <= s_from[a - 2]:
            p = 1.0
        elif a == 6 and s_from[a] < 4:
            p = 1.0
        elif a == 7 and s_from[a] < 1 and s_from[a - 1] == 4:
            p = 1.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_to = deepcopy(s_from)
            s_to[a] += 1
            return p, s_to
        else:
            return p, None

    def back_transition(self, s_to, a):
        # preconditions
        if s_to[a] > 0:
            if a == 0 and s_to[2] < 1:
                p = 1.0
            elif a == 1 and s_to[3] < 1:
                p = 1.0
            elif a in [2, 3] and s_to[a] > s_to[a + 2]:
                p = 1.0
            elif a in [6] and s_to[a + 1] < 1:
                p = 1.0
            elif a in [4, 5, 7]:
                p = 1.0
            else:
                p = 0.0
        else:
            p = 0.0

        # transition to next state
        if p == 1.0:
            s_from = deepcopy(s_to)
            s_from[a] -= 1
            return p, s_from
        else:
            return p, None

# ----------------------------------------- Training: Learn weights ------------------------------------------------- #
data = pd.read_csv("data/survey_data.csv")

for i in range(2,4):
    # initialize canonical task
    list = [[data['Q7_1'][i], data['Q7_2'][i], data['Q7_3'][i], data['Q7_4'][i], data['Q7_5'][i], data['Q7_6'][i]],
                [data['Q8_1'][i], data['Q8_2'][i], data['Q8_3'][i], data['Q8_4'][i], data['Q8_5'][i], data['Q8_6'][i]]]
    canonical_task = CanonicalTask(list)
    s_start = canonical_task.s_start
    actions = canonical_task.actions

    # list all states
    canonical_states = enumerate_states(s_start, actions, canonical_task.transition)

    # index of the terminal state
    terminal_idx = [len(canonical_states) - 1]

    # features for each state
    state_features = np.array(canonical_states)
    abstract_features = np.array([feature_vector(state, canonical_task.features) for state in canonical_states])


    # demonstrations
    canonical_demo = [[1, 0, 4, 3, 2, 5]]
    demo_trajectories = get_trajectories(canonical_states, canonical_demo, canonical_task.transition)

    print("Training ...")

    # using true features
    canonical_rewards_true, canonical_weights_true = maxent_irl(canonical_states,
                                                                actions,
                                                                canonical_task.transition,
                                                                canonical_task.back_transition,
                                                                state_features,
                                                                terminal_idx,
                                                                demo_trajectories,
                                                                optim, init)

    # using abstract features
    canonical_rewards_abstract, canonical_weights_abstract = maxent_irl(canonical_states,
                                                                        actions,
                                                                        canonical_task.transition,
                                                                        canonical_task.back_transition,
                                                                        abstract_features,
                                                                        terminal_idx,
                                                                        demo_trajectories,
                                                                        optim, init)

    print("Weights have been learned for the canonical task! Hopefully.")

    # ----------------------------------------- Verifying: Reproduce demo ----------------------------------------------- #

    qf_true, _, _ = value_iteration(canonical_states, actions, canonical_task.transition,
                                    canonical_rewards_true, terminal_idx)
    generated_sequence_true = rollout_trajectory(qf_true, canonical_states, canonical_demo, canonical_task.transition)

    qf_abstract, _, _ = value_iteration(canonical_states, actions, canonical_task.transition,
                                        canonical_rewards_abstract, terminal_idx)
    generated_sequence_abstract = rollout_trajectory(qf_abstract, canonical_states, canonical_demo, canonical_task.transition)

    print("\n")
    print("Canonical task:")
    print("       demonstration -", canonical_demo)
    print("    generated (true) -", generated_sequence_true)
    print("generated (abstract) -", generated_sequence_abstract)


    # ------------------------------------------ Testing: Predict complex ----------------------------------------------- #

    # initialize complex task
    complex_task = ComplexTask()
    s_start = complex_task.s_start
    actions = complex_task.actions

    # list all states
    complex_states = enumerate_states(s_start, actions, complex_task.transition)

    # index of the terminal state
    terminal_idx = [len(complex_states) - 1]

    # features for each state
    state_features = np.array([feature_vector(state, complex_task.features) for state in complex_states])

    # demonstrations
    complex_demo = [[6, 6, 6, 6, 0, 1, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 7]]
    demo_trajectories = get_trajectories(complex_states, complex_demo, complex_task.transition)

    # transfer rewards to complex task
    transfer_rewards_abstract = state_features.dot(canonical_weights_abstract)

    # rollout trajectory
    qf_abstract, _, _ = value_iteration(complex_states, actions, complex_task.transition,
                                        transfer_rewards_abstract, terminal_idx)
    predicted_sequence_abstract = rollout_trajectory(qf_abstract, complex_states, complex_demo, complex_task.transition)

    print("\n")
    print("Complex task:")
    print("       demonstration -", complex_demo)
    print("predicted (abstract) -", predicted_sequence_abstract)

    # ----------------------------------------- Verifying: Reproduce demo ----------------------------------------------- #

    # print("Training ...")
    # # inverse reinforcement learning
    # complex_rewards, weights = maxent_irl(complex_states,
    #                                       actions,
    #                                       complex_task.transition,
    #                                       complex_task.back_transition,
    #                                       state_features,
    #                                       terminal_idx,
    #                                       demo_trajectories,
    #                                       optim, init)
    #
    # print("Weights have been learned for the complex task! Hopefully.")
