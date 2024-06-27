import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from typing import List
from copy import deepcopy

import wandb

np.random.seed(42)


def server_averaging_mini_batches(
        global_model: nn.Module,
        client_models: List[nn.Module],
        global_optimizer: torch.optim.SGD,
):
    n_clients = len(client_models)

    with torch.no_grad():
        for client_model in client_models:
            for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                if global_param.grad is not None:
                    global_param.grad = global_param.grad + client_param.grad
                else:
                    global_param.grad = client_param.grad

                client_param.grad.zero_()

    global_optimizer.step()
    global_optimizer.zero_grad()

    global_state_dict = global_model.state_dict()

    for client_model in client_models:
        client_model.load_state_dict(global_state_dict)


def mini_batches_sgd(
        n_local_steps: int,
        lr: float,
        client_dataloaders: List[data.DataLoader],
        client_test_dataloaders: List[data.DataLoader],
        data_dim: int,
        writer: wandb.run
):
    n_clients = len(client_dataloaders)

    global_model = nn.Linear(data_dim, 1)
    client_models = [deepcopy(global_model) for _ in range(n_clients)]

    global_optimizer = torch.optim.SGD(global_model.parameters(), lr)

    criterion = nn.MSELoss()

    for global_step, client_data in enumerate(zip(*client_dataloaders)):
        for client_index in range(n_clients):
            loss = criterion(client_models[client_index](client_data[client_index][0]),
                             client_data[client_index][1]) / (n_local_steps * n_clients)
            loss.backward()

            writer.log({
                f"train_loss/client_{client_index}": loss.item(),
                "global_step": global_step
            })

        if (global_step + 1) % n_local_steps == 0:
            server_averaging_mini_batches(global_model,
                                          client_models,
                                          global_optimizer)

            # Evaluate
            with torch.no_grad():
                test_loss = 0
                for client_index in range(n_clients):
                    for test_data in client_test_dataloaders[client_index]:
                        test_loss += criterion(global_model(test_data[0]), test_data[1]).item()
                    test_loss = test_loss / len(client_test_dataloaders[client_index])

                test_loss = test_loss / n_clients
                writer.log({
                    "test_loss": test_loss,
                    "communication_round": (global_step + 1) // n_local_steps
                })
