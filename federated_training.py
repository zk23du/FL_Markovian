import torch

from markov_chain_utils import make_2_state_transition_matrix
from data_loader import MarkovianDataset, IIDDataset, TestDataset, make_data_loader
from local_sgd import local_sgd
from mini_batches_sgd import mini_batches_sgd
from fedprox import fedprox
from scaffold import scaffold

import numpy as np

from copy import deepcopy

import wandb

np.random.seed(42)


def federated_training():
    state_space_dim = 2
    data_dim = 10
    mu = 0.1
    stream_length = 10000
    batch_size = 32
    local_lr = 0.0001
    global_lr = 0.001

    n_clients = 10
    n_local_steps_list = [32, 64, 128]

    # common_transition_matrix = make_2_state_transition_matrix(1e-4)
    # eigen_values, _ = np.linalg.eigh(common_transition_matrix)
    # upper_bound_for_mixing_time = 1 / (1 - eigen_values[-2]) * 4 * (1 / state_space_dim)
    ps = [1e-10 * 10**i for i in range(10)]
    transition_matrices = [make_2_state_transition_matrix(p) for p in ps]
    upper_bounds_for_mixing_time = [1/p for p in ps]

    config = {
        "state_space_dim": state_space_dim,
        "data_dim": data_dim,
        "number_of_samples": stream_length,
        "batch_size": batch_size,
        "local_lr": local_lr,
        "global_lr": global_lr,
        "common_upper_bound_for_mixing_time": upper_bounds_for_mixing_time,
        "n_clients": n_clients,
    }

    client_data_space = [np.random.randn(state_space_dim, data_dim).astype(np.float32) for _ in range(n_clients)]
    # common_optimal = np.random.randn(data_dim, 1).astype(np.float32)
    # client_optimal_params = [deepcopy(common_optimal) for _ in range(n_clients)]
    client_optimal_params = [np.random.randn(data_dim, 1).astype(np.float32) for _ in range(n_clients)]

    client_markovian_datasets = [
        MarkovianDataset(
            client_data_space[i],
            transition_matrices[i],
            client_optimal_params[i],
            stream_length
        ) for i in range(n_clients)
    ]
    client_markovian_dataloaders = [
        make_data_loader(client_markovian_dataset, 1)
        for client_markovian_dataset in client_markovian_datasets
    ]

    client_iid_datasets = [
        IIDDataset(
            client_data_space[i],
            client_optimal_params[i],
            stream_length
        ) for i in range(n_clients)
    ]
    client_iid_dataloaders = [
        make_data_loader(client_iid_dataset, 1)
        for client_iid_dataset in client_iid_datasets
    ]

    client_test_datasets = [
        TestDataset(
            client_data_space[i],
            client_optimal_params[i]
        ) for i in range(n_clients)
    ]
    client_test_dataloaders = [
        make_data_loader(client_test_dataset, batch_size=batch_size)
        for client_test_dataset in client_test_datasets
    ]

    # run local sgd with markovian data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated Learning with Markovian data",
                name="local_sgd_diff_transition_matrix_diff_optimal_markovian_data",
                config=config
            )

            local_sgd(
                n_local_steps,
                stream_length,
                local_lr,
                client_markovian_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()

    # run local sgd with iid data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated Learning with Markovian data",
                name="local_sgd_diff_transition_matrix_diff_optimal_iid_data",
                config=config
            )

            local_sgd(
                n_local_steps,
                stream_length,
                local_lr,
                client_iid_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()

    # run mini batches sgd with markovian data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated learning with Markovian data",
                name="minibatches_sgd_diff_transition_matrix_diff_optimal_markovian_data",
                config=config
            )

            mini_batches_sgd(
                n_local_steps,
                global_lr,
                client_markovian_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()

    # run mini batches sgd with iid data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated learning with Markovian data",
                name="minibatches_sgd_diff_transition_matrix_diff_optimal_iid_data",
                config=config
            )

            mini_batches_sgd(
                n_local_steps,
                global_lr,
                client_iid_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()
    
# Run SCAFFOLD with Markovian data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated Learning with Markovian data",
                name="scaffold_diff_transition_matrix_diff_optimal_markovian_data",
                config=config
            )

            scaffold(
                n_local_steps,
                local_lr,
                client_markovian_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()

    # Run SCAFFOLD with IID data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated Learning with Markovian data",
                name="scaffold_diff_transition_matrix_diff_optimal_iid_data",
                config=config
            )

            scaffold(
                n_local_steps,
                local_lr,
                client_iid_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()
    # Run FedProx with Markovian data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated Learning with Markovian data",
                name="fedprox_diff_transition_matrix_diff_optimal_markovian_data",
                config=config
            )

            fedprox(
                n_local_steps,
                local_lr,
                mu,
                client_markovian_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()

    # Run FedProx with IID data
    for i in range(5):
        torch.manual_seed(i)

        for n_local_steps in n_local_steps_list:
            config["n_local_steps"] = n_local_steps

            run = wandb.init(
                project="Federated Learning with Markovian data",
                name="fedprox_diff_transition_matrix_diff_optimal_iid_data",
                config=config
            )

            fedprox(
                n_local_steps,
                local_lr,
                mu,
                client_iid_dataloaders,
                client_test_dataloaders,
                data_dim,
                run
            )

            run.finish()

if __name__ == "__main__":
    federated_training()
