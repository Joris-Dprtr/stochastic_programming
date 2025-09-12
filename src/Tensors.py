import torch
import numpy as np

def _moving_window(array, windows, window_size, window_step):
    """
    Create a moving window based on the steps that we want to forecast
    :param array: the flat array
    :param window_size: the N timesteps that we want to include in each window
    :param window_step: the step size
    :return: an array with dimensions taking the moving window into account
    """
    # Create an array of starting indices for each window
    start_indices = np.arange(0, windows * window_step, window_step)
    # Fill the array from the start index onwards
    index_array = start_indices[:, np.newaxis] + np.arange(window_size)
    # Apply the indexing on the original array
    return array[index_array]


def _scale(train, test, domain_min=None, domain_max=None):
    """
    MinMax scaling, fitting and transforming the train set, transforming the test set (with the train set fit)
    :param train: the train tensor
    :param test: the test tensor
    :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
    :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
    :return: returns the scaled train and test sets
    """
    minimum = domain_min if domain_min is not None else train.min()
    maximum = domain_max if domain_max is not None else train.max()

    denominator = maximum - minimum or 1e-8

    train = (train - minimum) / denominator
    test = (test - minimum) / denominator

    return train, test

class Tensors:
    def __init__(self,
                 data,
                 target: str,
                 past_features: list,
                 future_features: list,
                 lags: int,
                 forecast_period: int,
                 gap: int = 0,
                 forecast_gap: int = 0,
                 train_test_split: float = 0.8,
                 domain_min=None,
                 domain_max=None):
        """
        create tensors for use in pytorch
        :param data: the data (Pandas Dataframe)
        :param target: the target variable name
        :param past_features: A list of PAST feature names
        :param future_features: A list of FUTURE feature names
        :param lags: The number of lags to include (the input length)
        :param forecast_period: the number of hours to forecast
        :param gap: the gap between the lags and the forecast period
        :param forecast_gap: the gap between each forecast, can be negative
        :param train_test_split: the train test split as a float
        :param domain_min: a domain minimum (if known, otherwise train.min())
        :param domain_max: a domain maximum (if known, otherwise train.max())
        """

        self.data = data
        self.target = target
        self.past_features = past_features
        self.future_features = future_features
        self.lags = lags
        self.forecast_period = forecast_period
        self.gap = gap
        self.forecast_gap = forecast_gap
        self.train_test_split = train_test_split
        self.domain_min = domain_min
        self.domain_max = domain_max

    def create_tensor(self):
        """
        The method doing the tensor creation
        :return: tensors, split in a train and test set
        """

        start = self.lags + self.gap
        prediction_length = len(self.data) - start                                    # we can't forecast the first [lags] + [gap] timesteps
        divider = self.forecast_gap + self.forecast_period                            # The number of predictions is based on the forecast gap + forecast length

        # Check if the array is of the correct length
        left_over = prediction_length % divider
        if left_over:
            self.data = self.data[left_over:]
            prediction_length = len(self.data) - self.lags - self.gap

        predictions = (prediction_length - self.forecast_period) // divider + 1       # The number of samples

        train_len = round(predictions * self.train_test_split)
        test_len = predictions - train_len

        # Storage
        max_len = max(self.forecast_period, self.lags)
        feat_count = len(self.past_features) + len(self.future_features)

        X_train = torch.zeros(train_len, max_len, feat_count)                         # The dimensions of our X_Train tensor (Samples, period, features)
        X_test = torch.zeros(test_len, max_len, feat_count)                           # The dimensions of our X_Test tensor (Samples, period, features)

        # |Past features|
        for i, feat in enumerate(self.past_features):
            pad_left = max(0, self.forecast_period - self.lags)                       # Padding in case lags and forecast period are of different length
            train, test = self._feature_block(self.data[feat],                        # Take the correct slice of data and apply a moving window and scaling
                                              train_len,
                                              test_len,
                                              self.lags,
                                              divider,
                                              pad_left=pad_left,
                                              idx=i)
            X_train[:, :, i] = train                                                  # Inpute the feature in our overall tensor
            X_test[:, :, i] = test

        # |Future features|                                                           # Identical to the past features
        for j, feat in enumerate(self.future_features):
            pad_left = max(0, self.lags - self.forecast_period)
            train, test = self._feature_block(self.data[feat][start:],
                                              train_len,
                                              test_len,
                                              self.forecast_period,
                                              divider,
                                              pad_left=pad_left,
                                              idx=len(self.past_features) + j)
            X_train[:, :, len(self.past_features) + j] = train
            X_test[:, :, len(self.past_features) + j] = test

        # --- Target ---
        y_train, y_test = self._feature_block(self.data[self.target][start:],
                                              train_len,
                                              test_len,
                                              self.forecast_period,
                                              divider,
                                              idx=0)

        return X_train, X_test, y_train, y_test

    def _feature_block(self,
                       arr,
                       train_len,
                       test_len,
                       win_len,
                       divider,
                       pad_left=0,
                       pad_right=0,
                       idx=None):
        """
        Handles moving window, padding, scaling, and tensor conversion.
        :param arr: the current feature array
        :param train_len: the number of train samples
        :param test_len: the number of test samples
        :param win_len: number of lags or number of forecast period steps
        :param divider: To go from a flat to a correctly dimensioned array
        :param pad_left: padding in case the lags != forecast period
        :param pad_right: in case padding is done after the original array
        :param indx: The id of the feature, required for scaling
        """

        # Train/test slices
        train = np.array(arr[0: (train_len - 1) * divider + win_len])                 # The flat train array
        test = np.array(arr[(train_len * divider):
        (train_len + test_len) * divider + win_len])                                  # The flat test array

        # Reshape
        train = _moving_window(train, train_len, win_len, divider)                    # Obtain the correctly dimensioned train array
        test = _moving_window(test, test_len, win_len, divider)                       # Obtain the correctly dimensioned test array

        # Padding if mismatch between lags vs forecast
        if pad_left or pad_right:
            train = np.pad(train, ((0, 0), (pad_left, pad_right)))
            test = np.pad(test, ((0, 0), (pad_left, pad_right)))

        # Scaling based on domain knowledge, if available
        dmin = self.domain_min[idx] if isinstance(self.domain_min, list) else None
        dmax = self.domain_max[idx] if isinstance(self.domain_max, list) else None
        train, test = _scale(train, test, domain_min=dmin, domain_max=dmax)

        return torch.tensor(train, dtype=torch.float32), torch.tensor(test, dtype=torch.float32)