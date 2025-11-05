import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DeepAR(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout,
            distr_output="gaussian"):
        """
        Args:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            distr_output (str): 'gaussian' or 'beta'.
        """
        super(DeepAR, self).__init__()

        self.distr_output = distr_output.lower()

        # Core LSTM network
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        param_size = 2  # alpha, beta

        self.linear = nn.Linear(hidden_size, param_size)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, seq_len, input_size)
            h (tuple, optional): LSTM hidden state (h_0, c_0)

        Returns:
            params (tuple): Distribution parameters, e.g.
                            (mu, sigma) for Gaussian or (alpha, beta) for Beta.
            h (tuple): Updated LSTM states.
        """
        out, _ = self.lstm(x, None)  # (batch, seq_len, hidden_size)
        params = self.linear(out)  # (batch, seq_len, param_size)

        if self.distr_output == "gaussian":
            mu = params[..., 0]
            sigma = F.softplus(params[..., 1]) + 1e-6
            return mu, sigma

        elif self.distr_output == "beta":
            alpha = F.softplus(params[..., 0]) + 1e-6
            beta = F.softplus(params[..., 1]) + 1e-6
            return alpha, beta


class Training:

    def __init__(
            self,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            batch_size=32,
            learning_rate=0.001,
            ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_data = TensorDataset(X_test.to(self.device), y_test.to(self.device))
        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train(self):

        for epoch in range(self.epochs):
            batches = iter(self.train_loader)
            self.model.train()

            for input, output in batches:

                # Feed all but the last timestep (we predict next step)
                mu, sigma = self.model(input)  # (batch, seq_len-1)

                # Target = actual next value of the target variable
                target = output  # (batch, seq_len-1)

                # Negative log-likelihood loss for Gaussian
                # log N(y | mu, sigma) = -0.5 * [ log(2π) + 2log(sigma) + ((y - mu)/sigma)^2 ]
                nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + ((target - mu) ** 2) / (2 * sigma ** 2)
                loss = nll.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")