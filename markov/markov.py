import numpy as np


def markov_transition_matrix(
    sequence: np.ndarray,
    div_points: np.ndarray
):
    """Calculate markov transition probability matrix

    Args:
        sequence (np.ndarray): One-dimentional timeseries data.
        div_points (np.ndarray): Division points which define each state's boundary.
                                 The length must equal to state_num - 1.

    Returns:
        trans_prob_mat (np.ndarray): Transition probablity matrix. The probablity of transition
                                     from state i to j is given by trans_prob_mat[i, j]
        states_trans (np.ndarray): State transision timeseries data.
    """
    trans_prob_mat = np.zeros([len(div_points)-1, len(div_points)-1])
    states_trans = np.array([div_points < val for val in sequence]).sum(axis=1) - 1
    for i, j in zip(states_trans[:-1], states_trans[1:]):
        trans_prob_mat[i, j] += 1
    norms = trans_prob_mat.sum(axis=1)
    norms[norms == 0] = 1
    return (trans_prob_mat.T/norms).T, states_trans


def markov_likelihood(
    sequence: np.ndarray,
    div_points: np.ndarray,
    trans_prob_mat: np.ndarray,
):
    """Calculate transition likelihood based on pre-calculated
    markov transition probablity matrix.

    Args:


    Returns:

    """
    states_trans = np.array([div_points < val for val in sequence]).sum(axis=1) - 1
    likelihood = 1.
    for i, j in zip(states_trans[:-1], states_trans[1:]):
        # likelihood *= trans_prob_mat[i, j]
        likelihood += trans_prob_mat[i, j]
    return likelihood, states_trans
