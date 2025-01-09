from logging_project import logger

import torch
import torch.nn as nn

from src.data_prep import create_sequences

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def train_rnn_model(data,
                    input_size,
                    hidden_layer_size,
                    output_size,
                    num_layers,
                    lr,
                    num_epochs,
                    seq_len,
                    model_name):

    model = RNNModel(input_size=input_size,
                     hidden_size=hidden_layer_size,
                     output_size=output_size,
                     num_layers=num_layers)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X, y = create_sequences(data, seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X.unsqueeze(-1))
        loss = criterion(outputs, y)

        # Backward pass и оптимизация
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model, f'../models/{model_name}.pth')