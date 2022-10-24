import numpy as np
from scipy import stats
from vi import value_iteration
from copy import deepcopy


# ------------------------------------------------ IRL functions ---------------------------------------------------- #

def get_trajectories(task, demonstrations):
    """
    Convert sequence of actions to sequence of (s, a, s') tuples.
    Args:
        task: Task object with states and transitions
        demonstrations: List of action sequences

    Returns: Sequence of (s, a, s') tuples
    """
    trajectories = []
    for demo in demonstrations:
        s = task.states[0]
        trajectory = []
        for action in demo:
            p, sp = task.transition(s, action, most_likely=True)
            s_idx, sp_idx = task.states.index(s), task.states.index(sp)
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
                spl = task.transition_list(states[s_idx], a)
                for p, sp in spl:
                    sp_idx = task.states.index(sp)
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


def compute_expected_svf_using_rollouts(task, reward, max_iters=100):
    states, actions, terminal = task.states, task.actions, task.terminal_idx
    n_states, n_actions = len(states), len(actions)

    qf, vf, _ = value_iteration(states, actions, task.transition_list, reward, terminal)
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


def maxent_irl(task, s_features, trajectories, optim, omega_init, eps=1e-3):

    # states, actions = task.states, task.actions

    # number of actions and features
    n_states, n_features = s_features.shape

    # length of each demonstration
    demo_length = len(trajectories[0])

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(s_features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(task.states, trajectories)

    # gradient descent optimization
    omega = omega_init  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = s_features.dot(omega_old)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf_using_rollouts(task, reward)
        grad = e_features - s_features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute delta for convergence check
        delta = np.max(np.abs(omega_old - omega))
        # print(delta)

    # re-compute per-state reward and return
    return s_features.dot(omega), omega


# ----------------------------------------- Bayesian inference functions -------------------------------------------- #

def get_feature_count(state_features, trajectories):
    feature_counts = []
    for traj in trajectories:
        feature_count = deepcopy(state_features[traj[0][0]])
        for t in traj:
            feature_count += deepcopy(state_features[t[2]])
        feature_counts.append(feature_count)

    return feature_counts


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


def custom_likelihood(task, trajectories, qf):
    demos = np.array(trajectories)[:, :, 1]
    likelihood = []
    for demo in demos:
        p, _, _ = predict_trajectory(task, qf, [demo])
        likelihood.append(np.mean(p))

    return likelihood


# ------------------------------------------------ MDP functions ---------------------------------------------------- #

def random_predict_trajectory(task, demos, consider_options=True):
    """
    Randomly predict action at each step and compute prediction accuracy.
    """

    demo = demos[0]  # TODO: for demo in demos:
    s, available_actions = 0, demo.copy()

    generated_sequence, score = [], []
    for take_action in demo:
        candidates = []
        for a in available_actions:
            p, sp = task.transition(task.states[s], a)
            if sp:
                candidates.append(a)

        if not candidates:
            print("No available action in state", s)

        options = list(set(candidates))
        if consider_options:
            acc = options.count(take_action) / len(options)
        else:
            acc = float(np.random.choice(options) == take_action)
        score.append(acc)

        generated_sequence.append(take_action)
        p, sp = task.transition(task.states[s], take_action, most_likely=True)
        s = task.states.index(sp)
        available_actions.remove(take_action)

    return score, generated_sequence


def rollout_trajectory(task, qf, remaining_actions, start_state=0, state_info = False):
    """
    Execute remaining actions from state state based on qf (action values).
    """

    s = start_state
    available_actions = deepcopy(remaining_actions)
    generated_sequence, generated_trajectory = [], []
    while len(available_actions) > 0:
        max_action_val = -np.inf
        candidates = []
        for a in available_actions:
            p, sp = task.transition(task.states[s], a)
            if sp:
                if qf[s][a] > max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif qf[s][a] == max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        if not candidates:
            print(task.states[s])
        take_action = np.random.choice(candidates)
        generated_sequence.append(take_action)
        p, sp = task.transition(task.states[s], take_action)
        sp_idx = task.states.index(sp)
        generated_trajectory.append((s, take_action, sp_idx))
        if sp_idx != s:
            available_actions.remove(take_action)
        s = task.states.index(sp)

    if state_info:
        return generated_trajectory

    return generated_sequence


def predict_trajectory(task, qf, demos, sensitivity=0, consider_options=False):
    """
    Predict action at each step of user demonstration based on qf (action values).
    """

    demo = demos[0]  # TODO: for demo in demos:
    s, available_actions = 0, list(demo.copy())

    scores, predictions, options = [], [], []
    for step, take_action in enumerate(demo):

        max_action_val = -np.inf
        candidates, applicants = [], []
        for a in available_actions:
            p, sp = task.transition(task.states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1+sensitivity)*max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1-sensitivity)*max_action_val <= qf[s][a] <= (1+sensitivity)*max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        candidates = list(set(candidates))
        applicants = list(set(applicants))

        predictions.append(candidates)
        options.append(applicants)

        if consider_options:
            acc = options.count(take_action) / len(options)
        else:
            acc = float(np.random.choice(options) == take_action)
        scores.append(acc)

        # check inaccuracy
        if acc < 1.0:
            ro = rollout_trajectory(task, qf, available_actions, s)

            # confidence
            dp = ro.index(take_action)
            # c = dp / (qf[s][predict_action] - qf[s][take_action])
            print("Step:", step, "Score:", acc, "dp:", dp)

        p, sp = task.transition(task.states[s], take_action, most_likely=True)
        s = task.states.index(sp)
        available_actions.remove(take_action)

    return scores, predictions, options


# ------------------------------------------------- Contribution ---------------------------------------------------- #

def online_predict_trajectory(task, demos, task_trajectories, traj_likelihoods, weights, features, add_features,
                              samples, pref, optim, init, user_id, sensitivity=0, consider_options=True):
    """
    Predict action at each step of user demonstration based on qf (action values),
    and by updating the qf when the prediction is incorrect.
    """

    # assume the same starting state and available actions for all users
    demo = demos[0]
    s, available_actions = 0, list(demo.copy())

    # priors = np.ones(len(samples)) / len(samples)
    rewards = features.dot(weights)
    qf, _, _ = value_iteration(task.states, task.actions, task.transition_list, rewards, task.terminal_idx, delta=1e-3)

    up_weights, running_acc = [], []
    scores, predictions, options = [], [], []
    for step, take_action in enumerate(demo):

        # anticipate user action in current state
        max_action_val = -np.inf
        candidates, applicants = [], []
        for a in available_actions:
            p, sp = task.transition(task.states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        candidates = list(set(candidates))
        applicants = list(set(applicants))
        predictions.append(candidates)
        options.append(applicants)

        # calculate accuracy of prediction
        predict_action = np.random.choice(candidates)
        if consider_options:
            acc = candidates.count(take_action) / len(options)
        else:
            acc = float(predict_action == take_action)
        scores.append(acc)
        print("Predicted", candidates, "for", take_action)

        # update weights based on correct user action
        future_actions = deepcopy(available_actions)

        if acc < 1.0:  # or step == 0:

            n_iters = 10

            # confidence
            dfs = []
            for _ in range(n_iters):
                ro = rollout_trajectory(task, qf, future_actions, s)
                df = ro.index(take_action)
                # c = df / (qf[s][predict_action] - qf[s][take_action])
                dfs.append(df)
            dp = round(np.mean(dfs))

            print("Step:", step, "Score:", acc, "dp:", dp, "dq:", qf[s][predict_action] - qf[s][take_action])

            if dp > 3:

                # if "part" in pref:
                #     pref.remove("part")
                #     features = np.hstack((features, add_features[:, -2:-1]))
                #     print("Added new feature.")
                # elif "space" in pref:
                #     pref.remove("space")
                #     features = np.hstack((features, add_features[:, -1:]))
                #     print("Added new feature.")

                _, n_features = np.shape(features)
                prev_weights = init(n_features)
            else:
                prev_weights = deepcopy(weights)

            # approximate intended user action
            p, sp = task.transition(task.states[s], take_action, most_likely=True)
            current_user_demo = get_trajectories(task, [demo[:step+1]])[0]
            future_actions.remove(take_action)
            intended_trajectories = []
            for _ in range(n_iters):
                ro_traj = rollout_trajectory(task, qf, future_actions, task.states.index(sp), state_info=True)
                intended_traj = current_user_demo + ro_traj
                intended_trajectories.append(intended_traj)

            # intended_user_demo = [demo[:step] + [take_action] + ro]
            # intended_trajectories = get_trajectories(task, intended_user_demo)
            future_actions.append(take_action)

            # compute set from which user picks the intended action
            # all_complex_trajectories = [traj for traj in task_trajectories if all(traj[:step, 1] == demo[:step])]
            # all_intended_trajectories = [traj for traj in task_trajectories if all(traj[:step+1, 1] == demo[:step+1])]
            # likelihood_intention = boltzman_likelihood(features, all_intended_trajectories, prev_weights)
            # intention_idx = likelihood_intention.index(max(likelihood_intention))
            # intended_trajectories = [all_intended_trajectories[intention_idx]]

            # update weights
            # # bayesian approach
            # n_samples = 1000
            # new_samples, posterior = [], []
            # for n_sample in range(n_samples):
            #     weight_idx = np.random.choice(range(len(samples)), size=1, p=priors)[0]
            #     complex_weights = samples[weight_idx]
            #     # likelihood_all_traj, _ = boltzman_likelihood(features, task_trajectories, complex_weights)
            #     likelihood_all_traj = traj_likelihoods[weight_idx]
            #     # likelihood_user_demo = custom_likelihood(task, intended_trajectories, qf)
            #     likelihood_user_demo, r = boltzman_likelihood(features, intended_trajectories, complex_weights)
            #     likelihood_user_demo = likelihood_user_demo / np.sum(likelihood_all_traj)
            #     bayesian_update = (likelihood_user_demo * priors[n_sample])
            #
            #     new_samples.append(complex_weights)
            #     posterior.append(np.prod(bayesian_update))
            #
            # posterior = list(posterior / np.sum(posterior))
            # max_posterior = max(posterior)
            # weights = samples[posterior.index(max_posterior)]

            # max entropy approach
            # print("Re-learning weights ...")

            init_weights = prev_weights  # init(n_features)
            _, new_weights = maxent_irl(task, features, intended_trajectories, optim, init_weights, eps=1e-2)
            print("Updated weights from", weights, "to", new_weights)
            weights = deepcopy(new_weights)

            # compute policy for current estimate of weights
            rewards = features.dot(weights)
            qf, _, _ = value_iteration(task.states, task.actions, task.transition_list, rewards, task.terminal_idx,
                                       delta=1e-3)

        up_weights.append(weights)

        # priors = priors / np.sum(priors)
        p, sp = task.transition(task.states[s], take_action, most_likely=True)
        s = task.states.index(sp)
        available_actions.remove(take_action)

    return scores, predictions, options, up_weights, running_acc
