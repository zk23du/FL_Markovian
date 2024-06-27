import numpy as np


def make_symmetric_transition_matrix(
        dimension: int
) -> np.ndarray:
    """
    Generate a symmetric transition matrix
    :param dimension:
    :return: P: dimension x dimension
    """
    random_matrix = np.random.rand(dimension, dimension).astype(np.float32)
    symmetric_transition_matrix = 0.5 * (random_matrix + random_matrix.T)

    return symmetric_transition_matrix / np.sum(symmetric_transition_matrix, axis=1)[:, None]


def make_2_state_transition_matrix(
        p: float
) -> np.ndarray:
    """
    Generate a 2-state transition matrix
    :param p: probability of going from state 1 to state 2
    :return: the transition matrix
    """
    return np.array([
        [1 - p, p],
        [p, 1 - p]
    ])


def generate_markovian_data_stream(
        transition_matrix: np.ndarray,
        data: np.ndarray,
        stream_length: int,
) -> np.ndarray:
    """
    Generate a markovian stream of data with the given transition matrix
    :param transition_matrix: the transition matrix of the Markov chain
    :param data: the state space of the markov chain
    :param stream_length: number of data points to generate
    :return: the data stream
    """
    assert transition_matrix.shape == (data.shape[0], data.shape[0])

    data_dim = data.shape[1]
    state_space_dim = data.shape[0]

    current_state = np.random.choice(state_space_dim)
    data_stream = np.zeros((stream_length, data_dim), dtype=np.float32)
    data_stream[0] = data[current_state]

    for step in range(1, stream_length):
        next_state = np.random.choice(state_space_dim, p=transition_matrix[current_state])
        data_stream[step] = data[next_state]
        current_state = next_state

    return data_stream


def generate_iid_data_stream(
        data: np.ndarray,
        stream_length: int,
) -> np.ndarray:
    """
    Generate an i.i.d stream of data
    :param data:
    :param stream_length:
    :return: the data stream
    """
    state_space_dim, data_dim = data.shape

    data_stream = np.zeros((stream_length, data_dim), dtype=np.float32)

    for i in range(stream_length):
        state = np.random.choice(state_space_dim)
        data_stream[i] = data[state]

    return data_stream
