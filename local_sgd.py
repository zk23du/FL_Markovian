import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from typing import List
from copy import deepcopy

import wandb


np.random.seed(42)


# def sever_averaging(
#         global_model: nn.Module,
#         client_models: List[nn.Module]
# ):
#     global_state_dict = global_model.state_dict()

#     for key in global_state_dict.keys():
#         global_state_dict[key] = torch.stack([client_model.state_dict()[key] for client_model in client_models],
#                                              0).mean(0)

#     global_model.load_state_dict(global_state_dict)

#     for client_model in client_models:
#         client_model.load_state_dict(global_state_dict)

def server_averaging(global_model: nn.Module, client_models: List[nn.Module]):
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.stack([client_model.state_dict()[key] for client_model in client_models], 0).mean(0)
    global_model.load_state_dict(global_state_dict)
    for client_model in client_models:
        client_model.load_state_dict(global_state_dict)



def local_sgd(
        n_local_step: int,
        n_samples: int,
        lr: float,
        client_dataloader: List[data.DataLoader],
        client_test_dataloaders: List[data.DataLoader],
        data_dim: int,
        writer: wandb.run
):
    n_clients = len(client_dataloader)

    global_model = nn.Linear(data_dim, 1)
    client_models = [deepcopy(global_model) for _ in range(n_clients)]

    client_optimizers = [torch.optim.SGD(client_model.parameters(), lr=lr) for client_model in client_models]

    criterion = nn.MSELoss()

    for global_step, client_data in enumerate(zip(*client_dataloader)):
        for client_index in range(n_clients):
            loss = criterion(client_models[client_index](client_data[client_index][0]), client_data[client_index][1])
            loss.backward()
            writer.log({
                f"train_loss/client_{client_index}": loss.item(),
                "global_step": global_step
            })
            client_optimizers[client_index].step()
            client_optimizers[client_index].zero_grad()

        if (global_step + 1) % n_local_step == 0:
            # server averaging
            sever_averaging(global_model, client_models)

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
                    "communication_round": (global_step + 1) // n_local_step
                })
