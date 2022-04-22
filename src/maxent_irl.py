import numpy as np
from vi import value_iteration
from copy import deepcopy


def get_reward(state, curr_action, omega, s_feature, task):
    prev_action = state[-1]
    s_feature = np.append(s_feature, task.part_similarity[prev_action][curr_action])
    s_feature = np.append(s_feature, task.tool_similarity[prev_action][curr_action])

    return s_feature.dot(omega)


def get_trajectories(states, demonstrations, transition_function):
    trajectories = []
    for demo in demonstrations:
        s = states[0]
        trajectory = []
        for action in demo:
            p, sp = transition_function(s, action)
            s_idx, sp_idx = states.index(s), states.index(sp)
            trajectory.append((s_idx, action, sp_idx))
            s = sp
        trajectories.append(trajectory)

    return trajectories


def feature_expectation_from_trajectories(s_features, trajectories):
    n_states, n_features = s_features.shape

    fe = np.zeros(n_features)
    for t in trajectories:  # for each trajectory
        for s_idx, a, sp_idx in t:  # for each state in trajectory

            fe += s_features[sp_idx]  # sum-up features

    return fe / len(trajectories)  # average over trajectories


def initial_probabilities_from_trajectories(states, trajectories):
    n_states = len(states)
    prob = np.zeros(n_states)

    for t in trajectories:  # for each trajectory
        prob[t[0][0]] += 1.0  # increment starting state

    return prob / len(trajectories)  # normalize


def compute_expected_svf(task, p_initial, reward, max_iters, eps=1e-5):

    states, actions, terminal = task.states, task.actions, task.terminal_idx
    n_states, n_actions = len(states), len(actions)

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for i in range(2*n_states):
        za = np.zeros((n_states, n_actions))  # za: action partition function
        for s_idx in range(n_states):
            for a in actions:
                prob, sp = task.transition(states[s_idx], a)
                if sp:
                    sp_idx = task.states.index(sp)
                    if zs[sp_idx] > 0.0:
                        za[s_idx, a] += np.exp(reward[s_idx]) * zs[sp_idx]

        zs = za.sum(axis=1)
        zs[terminal] = 1.0

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, max_iters))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, max_iters):  # longest trajectory: n_states
        for sp_idx in range(n_states):
            parents = task.prev_states(states[sp_idx])
            if parents:
                for s in parents:
                    s_idx = states.index(s)
                    a = states[sp_idx][-1]
                    d[sp_idx, t] += d[s_idx, t - 1] * p_action[s_idx, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def compute_expected_svf_using_rollouts(task, reward, max_iters):
    states, actions, terminal = task.states, task.actions, task.terminal_idx
    n_states, n_actions = len(states), len(actions)

    qf, vf, _ = value_iteration(states, actions, task.transition, reward, terminal)
    svf = np.zeros(n_states)
    for _ in range(n_states):
        s_idx = 0
        svf[s_idx] += 1
        while s_idx not in task.terminal_idx:
            max_action_val = -np.inf
            candidates = []
            for a in task.actions:
                p, sp = task.transition(states[s_idx], a)
                if sp:
                    if qf[s_idx][a] > max_action_val:
                        candidates = [a]
                        max_action_val = qf[s_idx][a]
                    elif qf[s_idx][a] == max_action_val:
                        candidates.append(a)

            if not candidates:
                print("Error: No candidate actions from state", s_idx)

            take_action = np.random.choice(candidates)
            p, sp = task.transition(states[s_idx], take_action)
            s_idx = states.index(sp)
            svf[s_idx] += 1

    e_svf = svf/n_states

    return e_svf


def maxent_irl(task, s_features, trajectories, optim, init, eps=1e-3):

    # states, actions = task.states, task.actions

    # number of actions and features
    n_states, n_features = s_features.shape

    # length of each demonstration
    _, demo_length, _ = np.shape(trajectories)

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(s_features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(task.states, trajectories)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = s_features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf_using_rollouts(task, reward, demo_length)
        grad = e_features - s_features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute delta for convergence check
        delta = np.max(np.abs(omega_old - omega))
        # print(delta)

    # re-compute per-state reward and return
    return s_features.dot(omega), omega


def rollout_trajectory(qf, states, transition_function, remaining_actions, start_state=0):

    s = start_state
    available_actions = deepcopy(remaining_actions)
    generated_sequence = []
    while len(available_actions) > 0:
        max_action_val = -np.inf
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                if qf[s][a] > max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif qf[s][a] == max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        if not candidates:
            print(s)
        take_action = np.random.choice(candidates)
        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence


def predict_trajectory(qf, states, demos, transition_function, sensitivity=0, consider_options=False, qf_unknown=None):

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    action_pts = []
    predictions, scores = [], []
    decisions = []
    for take_action in demo:
        max_action_val = -np.inf
        max_action_val_new = -np.inf
        candidates = []
        candidates_new = []
        applicants = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

                if qf_unknown:
                    if qf_unknown[s][a] > (1 + sensitivity) * max_action_val_new:
                        candidates_new = [a]
                        max_action_val_new = qf_unknown[s][a]
                    elif (1 - sensitivity) * max_action_val_new <= qf_unknown[s][a] <= (1 + sensitivity) * max_action_val_new:
                        candidates_new.append(a)
                        max_action_val_new = qf_unknown[s][a]

        predictions.append(candidates)

        if len(candidates) > 1:
            predict_iters = 100
        elif len(candidates) == 1:
            predict_iters = 1
        else:
            print("Error: No candidate actions to pick from.")

        predict_score = []
        options = list(set(candidates))
        applicants = list(set(applicants))

        if len(applicants) > 1:
            decisions.append(True)
        else:
            decisions.append(False)

        if consider_options and (len(options) < len(applicants)):
            score = take_action in options
        else:
            for _ in range(predict_iters):
                predict_action = np.random.choice(options)
                predict_score.append(predict_action == take_action)
            score = np.mean(predict_score)
        scores.append(score)

        if qf_unknown:
            if candidates_new[0] == candidates[0]:
                action_pts.append(True)
            else:
                action_pts.append(False)

        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    if qf_unknown:
        return predictions, scores, action_pts
    else:
        return predictions, scores, decisions


def online_predict_trajectory(X, demos, all_traj, weights, features, samples, priors,
                              sensitivity=0, consider_options=False, qf_unknown=None):

    transition_function = X.transition
    states = X.states

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    action_pts = []
    predictions, scores = [], []
    decisions = []
    for step, take_action in enumerate(demo):
        rewards = features.dot(weights)
        qf, _, _ = value_iteration(X.states, X.actions, X.transition, rewards, X.terminal_idx)

        max_action_val = -np.inf
        max_action_val_new = -np.inf
        candidates = []
        candidates_new = []
        applicants = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

                if qf_unknown:
                    if qf_unknown[s][a] > (1 + sensitivity) * max_action_val_new:
                        candidates_new = [a]
                        max_action_val_new = qf_unknown[s][a]
                    elif (1 - sensitivity) * max_action_val_new <= qf_unknown[s][a] <= (1 + sensitivity) * max_action_val_new:
                        candidates_new.append(a)
                        max_action_val_new = qf_unknown[s][a]

        predictions.append(candidates)

        if len(candidates) > 1:
            predict_iters = 10
        elif len(candidates) == 1:
            predict_iters = 1
        else:
            print("Error: No candidate actions to pick from.")

        predict_score = []
        options = list(set(candidates))
        applicants = list(set(applicants))

        if len(applicants) > 1:
            decisions.append(True)
        else:
            decisions.append(False)

        if consider_options and (len(options) < len(applicants)):
            score = take_action in options
        else:
            for _ in range(predict_iters):
                predict_action = np.random.choice(options)
                predict_score.append(predict_action == take_action)
            score = np.mean(predict_score)
        scores.append(score)

        if qf_unknown:
            if candidates_new[0] == candidates[0]:
                action_pts.append(True)
            else:
                action_pts.append(False)

        future_actions = deepcopy(available_actions)
        if score < 0.55:
            prev_weights = deepcopy(weights)
            p, sp = transition_function(states[s], take_action)
            future_actions.remove(take_action)
            ro = rollout_trajectory(qf, states, transition_function, future_actions, states.index(sp))
            future_actions.append(take_action)
            complex_user_demo = [demo[:step] + [take_action] + ro]
            complex_trajectories = get_trajectories(states, complex_user_demo, transition_function)

            all_complex_trajectories = all_traj
            # for applicant in applicants:
            #     p, sp = transition_function(states[s], applicant)
            #     future_actions.remove(applicant)
            #     ro = rollout_trajectory(qf, states, transition_function, future_actions, states.index(sp))
            #     future_actions.append(applicant)
            #     complex_user_demo = [demo[:step] + [applicant] + ro]
            #     all_complex_trajectories = all_complex_trajectories + \
            #                                get_trajectories(states, complex_user_demo, transition_function)

            n_samples = 10
            new_samples = []
            posterior = []
            max_likelihood = - np.inf
            for _ in range(n_samples):
                # weight_idx = np.random.choice(range(len(samples)), size=1, p=priors)[0]
                # complex_weights = samples[weight_idx]
                u = np.random.uniform(0., 1., 3)
                d = np.sum(u)  # np.sum(u ** 2) ** 0.5
                complex_weights = u / d
                likelihood_all_trajectories, _ = boltzman_likelihood(features, all_complex_trajectories, complex_weights)
                likelihood_user_demo, r = boltzman_likelihood(features, complex_trajectories, complex_weights)
                likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_trajectories)

                new_samples.append(complex_weights)
                posterior.append(likelihood_user_demo[0])

                if likelihood_user_demo > max_likelihood:
                    max_likelihood = likelihood_user_demo
                    weights = complex_weights
                    max_reward = r

            posterior = posterior / np.sum(posterior)
            samples = deepcopy(new_samples)
            # priors = deepcopy(posterior)

            print("Updated weights from", prev_weights, "to", weights)

        # priors = priors / np.sum(priors)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    if qf_unknown:
        return predictions, scores, action_pts
    else:
        return predictions, scores, decisions


def actively_predict_trajectory(X, optim, init, qf, demos,
                                sensitivity=0, consider_options=False, qf_unknown=None):

    transition_function = X.transition
    states = X.states
    demo = demos[0]
    s, available_actions = 0, demo.copy()

    action_pts = []
    predictions, scores = [], []
    decisions = []
    for step, take_action in enumerate(demo):
        max_action_val = -np.inf
        max_action_val_new = -np.inf
        candidates = []
        candidates_new = []
        applicants = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

                if qf_unknown:
                    if qf_unknown[s][a] > (1 + sensitivity) * max_action_val_new:
                        candidates_new = [a]
                        max_action_val_new = qf_unknown[s][a]
                    elif (1 - sensitivity) * max_action_val_new <= qf_unknown[s][a] <= (1 + sensitivity) * max_action_val_new:
                        candidates_new.append(a)
                        max_action_val_new = qf_unknown[s][a]

        predictions.append(candidates)

        if len(candidates) > 1:
            predict_iters = 100
        elif len(candidates) == 1:
            predict_iters = 1
        else:
            print("Error: No candidate actions to pick from.")

        predict_score = []
        options = list(set(candidates))
        applicants = list(set(applicants))

        if len(applicants) > 1:
            decisions.append(True)
        else:
            decisions.append(False)

        if consider_options and (len(options) < len(applicants)):
            score = take_action in options
        elif len(options) > 0:
            for _ in range(predict_iters):
                predict_action = np.random.choice(options)
                predict_score.append(predict_action == take_action)
            score = np.mean(predict_score)
        else:
            score = 0
        scores.append(score)

        if qf_unknown:
            if candidates_new[0] == candidates[0]:
                action_pts.append(True)
            else:
                action_pts.append(False)

        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

        if score < 0.8:
            future_actions = deepcopy(available_actions)
            ro = rollout_trajectory(qf, states, transition_function, future_actions, s)
            complex_user_demo = [demo[:step+1] + ro]
            complex_trajectories = get_trajectories(states, complex_user_demo, transition_function)

            print("Training ...")

            # using abstract features
            abstract_features = np.array([X.get_features(state) for state in X.states])
            norm_abstract_features = abstract_features / np.linalg.norm(abstract_features, axis=0)
            complex_rewards_abstract, complex_weights_abstract = maxent_irl(X, norm_abstract_features,
                                                                            complex_trajectories,
                                                                            optim, init)

            qf_transfer, _, _ = value_iteration(X.states, X.actions, X.transition, complex_rewards_abstract,
                                                X.terminal_idx)
            qf = qf_transfer

    if qf_unknown:
        return predictions, scores, action_pts
    else:
        return predictions, scores, decisions


def random_trajectory(states, demos, transition_function):
    """
    random predicted trajectory
    """

    demo = demos[0]
    s, available_actions = 0, demo.copy()

    generated_sequence, score = [], []
    for take_action in demo:
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                candidates.append(a)

        if not candidates:
            print(s)

        options = list(set(candidates))
        predict_action = np.random.choice(options)
        if take_action in options:
            acc = 1/len(options)
        else:
            acc = 0.0
        score.append(acc)

        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence, score


def boltzman_likelihood(state_features, trajectories, weights, rationality=0.99):
    n_states, n_features = np.shape(state_features)
    likelihood, rewards = [], []
    for traj in trajectories:
        feature_count = deepcopy(state_features[traj[0][0]])
        for t in traj:
            feature_count += deepcopy(state_features[t[2]])
        total_reward = rationality * weights.dot(feature_count)
        rewards.append(total_reward)
        likelihood.append(np.exp(total_reward))

    return likelihood, rewards

