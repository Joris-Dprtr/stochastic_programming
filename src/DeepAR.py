import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import pyplot as plt


def _beta_nll(alpha, beta, x):
    """
    Compute Negative Log-Likelihood for Beta distribution

    Parameters:
    :param alpha: Tensor - predicted alpha > 0
    :param beta: Tensor - predicted beta > 0
    :param x: Tensor - observed values in (0, 1)
    :return Tensor - NLL value
    """

    eps = 1e-6
    x = x.clamp(eps, 1 - eps)


    # Compute log-likelihood for data
    log_likelihood = ((alpha - 1) * torch.log(x) +
                      (beta - 1) * torch.log(1 - x)).sum()

    # Normalization term using log Gamma (torch.lgamma is log Gamma)
    normalization = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)

    # Negative log-likelihood
    nll = - (log_likelihood - x.shape[0] * normalization)
    return nll


def _gaus_nll(mu, sigma, x):
    """
    Compute Negative Log-Likelihood for Gaussian distribution

    :param mu: Tensor - predicted mu
    :param sigma: Tensor - predicted sigma > 0
    :param x: Tensor - observed value
    :return: Tensor - NLL value
    """
    nll = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + ((x - mu) ** 2) / (2 * sigma ** 2)
    return nll


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

    def fit(self, verbose=False):

        avg_train_error = []
        avg_test_error = []
        state_dict_list = []

        for epoch in range(self.epochs):

            num_train_batches = 0
            num_test_batches = 0
            total_loss = 0
            total_test_loss = 0

            batches = iter(self.train_loader)
            self.model.train()

            for input, output in batches:

                if self.model.distr_output == 'gaussian':
                    mu, sigma = self.model(input)  # (batch, seq_len-1)
                    nll = _gaus_nll(mu, sigma, output)

                else:
                    alpha, beta = self.model(input)
                    nll = _beta_nll(alpha, beta, output)

                loss = nll.mean()

                total_loss += float(loss.detach())
                num_train_batches += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()

            with torch.inference_mode():

                test_batches = iter(self.test_loader)

                for input, output in test_batches:

                    if self.model.distr_output == 'gaussian':
                        mu, sigma = self.model(input)  # (batch, seq_len-1)
                        nll = _gaus_nll(mu, sigma, output)

                    else:
                        alpha, beta = self.model(input)
                        nll = _beta_nll(alpha, beta, output)

                    test_loss = nll.mean()

                    total_test_loss += float(test_loss.detach())
                    num_test_batches += 1

            avg_train_error.append(total_loss / num_train_batches)
            avg_test_error.append(total_test_loss / num_test_batches)

            state_dict_list.append(self.model.state_dict())

            if epoch % 5 == 0:
                print('Step {}: Average train loss: {:.4f} | Average test loss: {:.4f}'.format(epoch,
                                                                                               avg_train_error[epoch],
                                                                                               avg_test_error[epoch]))

        argmin_test = avg_test_error.index(min(avg_test_error))

        print('Best Epoch: ' + str(argmin_test))

        if verbose:
            plt.plot(avg_train_error, label='train error')
            plt.plot(avg_test_error, label='test error')
            plt.legend()

        return state_dict_list, argmin_test
