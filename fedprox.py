
from local_sgd import server_averaging 
def fedprox(
        n_local_steps: int,
        lr: float,
        mu: float,
        client_dataloaders: List[data.DataLoader],
        client_test_dataloaders: List[data.DataLoader],
        data_dim: int,
        writer: wandb.run
):
    n_clients = len(client_dataloaders)

    global_model = nn.Linear(data_dim, 1)
    client_models = [deepcopy(global_model) for _ in range(n_clients)]

    client_optimizers = [torch.optim.SGD(client_model.parameters(), lr=lr) for client_model in client_models]

    criterion = nn.MSELoss()

    for global_step, client_data in enumerate(zip(*client_dataloaders)):
        for client_index in range(n_clients):
            client_model = client_models[client_index]
            client_optimizer = client_optimizers[client_index]

            client_model.train()
            for step in range(n_local_steps):
                data, target = client_data[client_index]
                client_optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                
                # Proximal term
                proximal_term = 0
                for name, param in client_model.named_parameters():
                    proximal_term += (param - global_model.state_dict()[name]).norm(2).item()

                loss += (mu / 2) * proximal_term
                loss.backward()

                client_optimizer.step()

                writer.log({
                    f"train_loss/client_{client_index}": loss.item(),
                    "global_step": global_step * n_local_steps + step
                })

        # Server aggregation
        server_averaging(global_model, client_models)

        # Evaluate
        with torch.no_grad():
            test_loss = 0
            for client_index in range(n_clients):
                client_test_dataloader = client_test_dataloaders[client_index]
                for test_data, test_target in client_test_dataloader:
                    test_loss += criterion(global_model(test_data), test_target).item()

            test_loss /= (len(client_test_dataloaders) * len(client_test_dataloaders[0]))
            writer.log({
                "test_loss": test_loss,
                "communication_round": global_step
            })
