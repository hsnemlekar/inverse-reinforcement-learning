import numpy as np
from itertools import product


def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:
        for s in t.states():
            fe += features[s]

    return fe / len(trajectories)


def local_action_probabilities(p_transition, terminal, reward):
    n_states, _, n_actions = p_transition.shape

    er = np.exp(reward)
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # initialize at terminal states
    zs = np.zeros(n_states)
    for s in terminal:
        zs[s] = 1.0

    # perform backward pass
    # This does not converge, instead we iterate a fixed number of steps. The
    # number of steps is chosen to reflect the maximum steps required to
    # guarantee propagation from any state to any other state and back in an
    # arbitrary MDP defined by p_transition.
    for _ in range(2 * n_states):
        za = np.array([np.multiply(er, np.dot(p[a], zs)) for a in range(n_actions)]).T
        zs = np.sum(za, axis=1)

    # compute local action probabilities
    return za / zs[:, None]


def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
    n_states, _, n_actions = p_transition.shape

    # set-up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # 'fix' our policy to allow for convergence
    # we will _never_ leave any terminal state
    p_action = np.copy(p_action)
    p_action[terminal, :] = 0.0

    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [np.dot(p_transition[a].T, np.multiply(p_action[:, a], d)) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    return d