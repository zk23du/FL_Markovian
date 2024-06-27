import torch.utils.data as data
import numpy as np

from markov_chain_utils import (generate_markovian_data_stream,
                                generate_iid_data_stream)


class MarkovianDataset(data.Dataset):
    def __init__(
            self,
            data_space: np.ndarray,
            transition_matrix: np.ndarray,
            optimal_params: np.ndarray,
            n_samples: int,
    ):
        super(MarkovianDataset, self).__init__()

        assert transition_matrix.shape == (data_space.shape[0], data_space.shape[0])
        assert optimal_params.shape == (data_space.shape[1], 1)

        self.data_space = data_space
        self.transition_matrix = transition_matrix
        self.n_samples = n_samples
        self.optimal_params = optimal_params

        self.state_space_dim, self.data_dim = data_space.shape

        self.data_stream = generate_markovian_data_stream(
            transition_matrix=self.transition_matrix,
            data=self.data_space,
            stream_length=self.n_samples,
        )
        self.labels = np.dot(self.data_stream, self.optimal_params)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data_stream[idx], self.labels[idx]


class IIDDataset(data.Dataset):
    def __init__(
            self,
            data_space: np.ndarray,
            optimal_params: np.ndarray,
            n_samples: int,
    ):
        super(IIDDataset, self).__init__()

        assert optimal_params.shape == (data_space.shape[1], 1)

        self.data_space = data_space
        self.optimal_params = optimal_params
        self.n_samples = n_samples

        self.data_stream = generate_iid_data_stream(
            data=data_space,
            stream_length=n_samples
        )
        self.labels = np.dot(self.data_stream, self.optimal_params)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data_stream[idx], self.labels[idx]


class TestDataset(data.Dataset):
    def __init__(
            self,
            data_space: np.ndarray,
            optimal_params: np.ndarray,
    ):
        super(TestDataset, self).__init__()

        assert optimal_params.shape == (data_space.shape[1], 1)

        self.data_space = data_space
        self.optimal_params = optimal_params
        self.labels = np.dot(self.data_space, self.optimal_params)

    def __len__(self):
        return self.data_space.shape[0]

    def __getitem__(self, idx):
        return self.data_space[idx], self.labels[idx]


def make_data_loader(
        dataset: data.Dataset,
        batch_size: int
):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
