import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from IPython.display import clear_output

def save_model(model, name):
    """
    Saves the state dictionary using torch
    :param name: name of the file
    :param model: the model for which we want to save the state dictionary
    """
    torch.save(model.state_dict(), '../models/' + str(name))


def _rescale(values, scaler):
    rescaled_values = values * (scaler[1] - scaler[0]) + scaler[0]
    return rescaled_values


class Training:

    def __init__(
            self,
            model,
            cvx_layer,
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            epochs,
            T,
            bat_cap,
            batch_size=32,
            learning_rate=0.001,
            criterion=torch.nn.MSELoss(),
            lr_decay=None
    ):
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
        :param criterion: the criterion by which to evaluate the performance (i.e. the loss function)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        train_data = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_data = TensorDataset(X_test.to(self.device), y_test.to(self.device))
        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)
        self.scaler = scaler

        self.model = model
        self.cvx = cvx_layer
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.epochs = epochs

        self.lr_decay = lr_decay
        if lr_decay == 'Linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.epochs)
        elif lr_decay == 'Exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        elif lr_decay == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.T = T
        self.bat_cap = bat_cap

    def fit(self, verbose=0):
        """
        The training loop itself with error handling for cvx layer issues.
        :return: state_dict_list: the state dictionary for each of the epochs, argmin_test: the best epoch
        """
        avg_train_error = []
        avg_train_mse_pv = []
        avg_train_mse_load = []
        avg_test_error = []
        avg_test_mse_pv = []
        avg_test_mse_load = []
        state_dict_list = []
        try:
            for epoch in range(self.epochs):
                num_train_batches = 0
                num_test_batches = 0
                total_loss = 0
                total_mse_pv = 0
                total_mse_load = 0
                total_loss_test = 0
                total_mse_test_pv = 0
                total_mse_test_load = 0

                batches = iter(self.train_loader)
                self.model.train()

                for input, output in batches:
                    pv_train, load_train, y_train = self.model(input[:, :, [0,2,3,4]],
                                               input[:, :, [1,3,4]],
                                               input[:, -self.T:, -3],
                                               input[:, -self.T:, -2],
                                               input[:, -1, -1])

                    y_train = self.cvx(_rescale(output[:, :, 0], self.scaler[0]),
                                       _rescale(output[:, :, 1], self.scaler[1]),
                                       input[:, -self.T:, -3],
                                       input[:, -self.T:, -2],
                                       input[:, -1, -1],
                                       y_train[-2],
                                       y_train[-1])

                    prediction = (torch.bmm(y_train[0].unsqueeze(1), input[:, -self.T:, -3].unsqueeze(-1)) -
                                  torch.bmm(y_train[1].unsqueeze(1), input[:, -self.T:, -2].unsqueeze(-1)))
                    prediction = prediction.squeeze([1, 2])

                    mse_pv = self.criterion(pv_train, output[:, :, 0])
                    mse_load = self.criterion(load_train, output[:, :, 1])
                    regret = self.criterion(prediction, output[:, -1, 2])

                    total_loss += float(regret)
                    total_mse_pv += float(mse_pv)
                    total_mse_load += float(mse_load)

                    num_train_batches += 1

                    self.optimizer.zero_grad()
                    regret.backward()
                    self.optimizer.step()

                self.model.eval()

                with torch.inference_mode():
                    test_batches = iter(self.test_loader)

                    for input, output in test_batches:
                        pv_test, load_test, y_test = self.model(input[:, :, [0,2,3,4]],
                                                                input[:, :, [1,3,4]],
                                                                input[:, -self.T:, -3],
                                                                input[:, -self.T:, -2],
                                                                input[:, -1, -1])

                        y_test = self.cvx(_rescale(output[:, :, 0], self.scaler[0]),
                                          _rescale(output[:, :, 1], self.scaler[1]),
                                          input[:, -self.T:, -3],
                                          input[:, -self.T:, -2],
                                          input[:, -1, -1],
                                          y_test[-2],
                                          y_test[-1])

                        prediction = (torch.bmm(y_test[0].unsqueeze(1), input[:, -self.T:, -3].unsqueeze(-1)) -
                                      torch.bmm(y_test[1].unsqueeze(1), input[:, -self.T:, -2].unsqueeze(-1)))
                        prediction = prediction.squeeze([1, 2])

                        mse_pv = self.criterion(pv_test, output[:, :, 0])
                        mse_load = self.criterion(load_test, output[:, :, 1])
                        regret = self.criterion(prediction, output[:, -1, 2])

                        total_loss_test += float(regret)
                        total_mse_test_pv += float(mse_pv)
                        total_mse_test_load += float(mse_load)

                        num_test_batches += 1

                avg_train_error.append(total_loss / num_train_batches)
                avg_train_mse_pv.append(total_mse_pv / num_train_batches)
                avg_train_mse_load.append(total_mse_load / num_train_batches)

                avg_test_error.append(total_loss_test / num_test_batches)
                avg_test_mse_pv.append(total_mse_test_pv / num_test_batches)
                avg_test_mse_load.append(total_mse_test_load / num_test_batches)

                # state_dict_list.append(self.model.state_dict())
                state_dict_list.append({k: v.clone().detach() for k, v in self.model.state_dict().items()})

                if epoch % 5 == 0 and verbose >= 2:
                    print('Step {}:\n'
                          'Average train loss: {:.4f} | Average test loss: {:.4f}\n'
                          '   MSE PV: {:.4f} | {:.4f}\n'
                          '   MSE LOAD: {:.4f} | {:.4f}\n'
                          .format(epoch,
                                  avg_train_error[epoch], avg_test_error[epoch],
                                  avg_train_mse_pv[epoch], avg_test_mse_pv[epoch],
                                  avg_train_mse_load[epoch], avg_test_mse_load[epoch]))


                if self.lr_decay is not None:
                    self.scheduler.step()

        except Exception as e:
            print(f"Training interrupted due to error: {e}")
            print("Returning progress up to this point...")

        argmin_test = avg_test_error.index(min(avg_test_error)) if avg_test_error else -1

        print('Best Epoch: ' + str(argmin_test))

        if verbose >= 1:

            plt.plot(avg_train_error, label='train error')
            plt.plot(avg_test_error, label='test error')
            plt.plot(avg_train_mse_pv, label='train MSE PV')
            plt.plot(avg_test_mse_pv, label='test MSE PV')
            plt.plot(avg_train_mse_load, label='train MSE LOAD')
            plt.plot(avg_test_mse_load, label='test MSE LOAD')
            plt.title('Error')
            plt.legend()

            plt.tight_layout()
            plt.show()

        return state_dict_list, argmin_test
