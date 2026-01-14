import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer


class CVXLayer(nn.Module):
    def __init__(self,
                 problem,
                 parameters: list,
                 variables: list):
        super().__init__()

        self.cvxpylayer = CvxpyLayer(problem,
                                     parameters=parameters,
                                     variables=variables)

    def forward(self, *x):
        y = self.cvxpylayer(*x)

        return y


class LSTMOPT_DUAL(nn.Module):
    def __init__(
            self,
            input_size_1,
            input_size_2,
            hidden_size,
            num_layers,
            output_size,
            dropout,
            problem,
            parameters,
            variables,
            scaler_1,
            scaler_2,
            ):
        """
        Simple OLD_LSTM model made in pytorch
        :param input_size: the size of the input (based on the lags provided)
        :param hidden_size: the hidden layer sizes
        :param num_layers: the number of layers in the OLD_LSTM (each of size hidden_size)
        :param output_size: the forecast window (f.e. 24 means 'forecast 24 hours')
        """
        super().__init__()

        self.lstm_1 = nn.LSTM(input_size_1, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.linear_1 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.lstm_2 = nn.LSTM(input_size_2, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.cvxlayer = CVXLayer(problem, parameters, variables)

        self.scaler_1 = scaler_1
        self.scaler_2 = scaler_2

    def forward(self, input_1, input_2, *parameters):
        """
        Forward method for the OLD_LSTM layer. I.e. how input gets processed
        :param input: the input tensor
        :return: output tensor
        """
        hidden_1, _ = self.lstm_1(input_1, None)
        if hidden_1.dim() == 2:
            hidden_1 = hidden_1[-1, :]
        else:
            hidden_1 = hidden_1[:, -1, :]
        output_1 = self.linear_1(hidden_1)
        rescaled_output_1 = output_1 * (self.scaler_1[1] - self.scaler_1[0]) + self.scaler_1[0]

        hidden_2, _ = self.lstm_2(input_2, None)
        if hidden_2.dim() == 2:
            hidden_2 = hidden_2[-1, :]
        else:
            hidden_2 = hidden_2[:, -1, :]
        output_2 = self.linear_2(hidden_2)
        rescaled_output_2 = output_2 * (self.scaler_2[1] - self.scaler_2[0]) + self.scaler_2[0]

        opt_output = self.cvxlayer(rescaled_output_1, rescaled_output_2 , *parameters)

        return output_1, output_2, opt_output
