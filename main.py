import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import SGD

from markov_chain_utils import make_symmetric_transition_matrix, make_2_state_transition_matrix
from data_loader import MarkovianDataset, IIDDataset, TestDataset, make_data_loader

from copy import deepcopy


def test_centralized(name: str):
    state_space_dim = 2
    data_dim = 10
    stream_length = 500
    batch_size = 1
    lr = 0.001

    transition_matrix = make_2_state_transition_matrix(1e-4)
    eigen_values, _ = np.linalg.eigh(transition_matrix)
    upper_bound_for_mixing_time = 1/(1 - eigen_values[-2]) * 4 * (1/state_space_dim)

    config = {
        "state_space_dim": state_space_dim,
        "data_dim": data_dim,
        "number_of_samples": stream_length,
        "batch_size": batch_size,
        "lr": lr,
        "upper_bound_for_mixing_time": upper_bound_for_mixing_time
    }

    wandb.init(
        project="SGD with markovian data",
        name=name,
        config=config
    )
    wandb.define_metric("*", step_metric="global_step")

    data_space = np.random.randn(state_space_dim, data_dim).astype(np.float32)
    optimal_params = np.random.randn(data_dim, 1).astype(np.float32)

    markovian_dataset = MarkovianDataset(
        data_space,
        transition_matrix,
        optimal_params,
        stream_length
    )
    markovian_dataloader = make_data_loader(markovian_dataset, batch_size=batch_size)

    iid_dataset = IIDDataset(
        data_space,
        optimal_params,
        stream_length
    )
    iid_dataloader = make_data_loader(iid_dataset, batch_size=batch_size)

    test_dataset = TestDataset(
        data_space,
        optimal_params
    )
    test_dataloader = make_data_loader(test_dataset, batch_size=batch_size)

    linear_model_markovian = nn.Linear(data_dim, 1)
    linear_model_iid = deepcopy(linear_model_markovian)
    criterion = nn.MSELoss()

    optimizer_markovian = SGD(linear_model_markovian.parameters(), lr=lr)
    optimizer_iid = SGD(linear_model_iid.parameters(), lr=lr)

    for i, (markovian_data, iid_data) in enumerate(zip(markovian_dataloader, iid_dataloader)):
        markovian_loss = criterion(linear_model_markovian(markovian_data[0]), markovian_data[1])
        markovian_loss.backward()
        wandb.log({"train_loss/markovian_data": markovian_loss.item(), "global_step": i})
        optimizer_markovian.step()
        optimizer_markovian.zero_grad()

        iid_loss = criterion(linear_model_iid(iid_data[0]), iid_data[1])
        iid_loss.backward()
        wandb.log({"train_loss/iid_data": iid_loss.item(), "global_step": i})
        optimizer_iid.step()
        optimizer_iid.zero_grad()

        # evaluate
        with torch.no_grad():
            test_loss_markovian = 0
            test_loss_iid = 0
            for test_data in test_dataloader:
                test_loss_markovian += criterion(linear_model_markovian(test_data[0]), test_data[1]).item()
                test_loss_iid += criterion(linear_model_iid(test_data[0]), test_data[1]).item()

            test_loss_markovian = test_loss_markovian / len(test_dataloader)
            test_loss_iid = test_loss_iid / len(test_dataloader)
            wandb.log({"test_loss/markovian": test_loss_markovian, "global_step": i})
            wandb.log({"test_loss/iid": test_loss_iid, "global_step": i})

    wandb.finish()


if __name__ == '__main__':
    for i in range(5):
        torch.manual_seed(i)
        np.random.seed(i)
        test_centralized(name=f"centralized_with_2_states_{i}")
