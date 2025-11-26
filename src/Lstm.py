import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


class LSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
            dropout):
        """
        Simple LSTM model made in pytorch
        :param input_size: the size of the input (based on the lags provided)
        :param hidden_size: the hidden layer sizes
        :param num_layers: the number of layers in the LSTM (each of size hidden_size)
        :param output_size: the forecast window (f.e. 24 means 'forecast 24 hours')
        :param dropout: the dropout parameter used for training, to avoid overfitting
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        """
        Forward method for the LSTM layer. I.e. how input gets processed
        :param input: the input tensor
        :return: output tensor
        """
        hidden, _ = self.lstm(input, None)
        if hidden.dim() == 2:
            hidden = hidden[-1, :]
        else:
            hidden = hidden[:, -1, :]
        output = self.linear(hidden)

        return output

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
            criterion=torch.nn.MSELoss()):
        """
        The training class for the pytorch model
        :param model: The model that we train
        :param X_train: the tensor with training values for X
        :param y_train: the tensor with training values for y
        :param X_test: the tensor with test values for X
        :param y_test: the tensor with test values for y
        :param epochs: the number of epochs that we wish to train for
        :param batch_size: the batch size before going through backpropagation
        :param learning_rate: the learning rate
        :param criterion: the criterion by which to evaluate the performance
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_data = TensorDataset(X_test.to(self.device), y_test.to(self.device))
        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)

        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def fit(self, verbose=False):
        """
        The training loop itself
        :return: state_dict_list: the state dictionary for each of the epochs, argmin_test: the best epoch
        """
        avg_train_error = []
        avg_test_error = []
        state_dict_list = []

        for epoch in range(self.epochs):
            num_train_batches = 0
            num_test_batches = 0
            total_loss = 0
            total_loss_test = 0
            batches = iter(self.train_loader)
            self.model.train()

            # Do a forward and backwards pass for every batch
            for input, output in batches:
                prediction = self.model(input)                                            # Prediction based on current model
                output = output.squeeze()                                                 # The correct values
                loss = self.criterion(prediction, output)                                 # Evaluate prediction

                total_loss += float(loss.detach())
                num_train_batches += 1

                self.optimizer.zero_grad()
                loss.backward()                                                           # Take a backwards step based on the current batch
                self.optimizer.step()                                                     # Update the state dictionary

            self.model.eval()                                                             # Now evaluate the performance on the test set

            with torch.inference_mode():

                test_batches = iter(self.test_loader)

                for input, output in test_batches:
                    prediction = self.model(input)
                    output = output.squeeze()
                    test_loss = self.criterion(prediction, output)

                    total_loss_test += float(test_loss.detach())
                    num_test_batches += 1

            avg_train_error.append(total_loss / num_train_batches)
            avg_test_error.append(total_loss_test / num_test_batches)

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