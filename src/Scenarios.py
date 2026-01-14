import numpy as np

import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

class Scenario_tool:

    HOURS_IN_DAY = 24

    def __init__(self,
                 model,
                 model_distribution,
                 data,
                 days
                 ):

        self.model = model
        self.model_distribution = model_distribution
        self.data = data
        self.days = days
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def scenario_generation(self, num_scenarios, verbose = 0):

        days_in_hours = range(0, self.data.shape[0], self.HOURS_IN_DAY)
        scenario_data = torch.empty((self.days, self.HOURS_IN_DAY, num_scenarios), device=self.device)

        self.model.eval()

        with torch.no_grad():
            for day in days_in_hours:
                day_nr = int(day / 24)

                # --- PRIMING PHASE (First hour) ---

                # 1. Run full history up to time t
                # (This remains sequential, as it's only one sequence)
                start_forecast = 1
                end_forecast = 24

                _, (h_initial_single, c_initial_single) = self.model.lstm(
                    self.data[day, :, :].to(self.device)
                )

                # 2. Calculate initial parameters for t+1
                # This requires running the last step again with the state h_t, c_t
                last_output, _ = self.model.lstm(
                    self.data[day, -1:, :].to(self.device),
                    (h_initial_single, c_initial_single)
                )
                params_t_plus_1 = self.model.linear(last_output.squeeze(0).squeeze(0))

                if self.model_distribution == 'gaussian':
                    mu_t1 = params_t_plus_1[0]
                    sigma_t1 = F.softplus(params_t_plus_1[1]) + 1e-6
                    mu_t1_batch = mu_t1.repeat(num_scenarios)
                    sigma_t1_batch = sigma_t1.repeat(num_scenarios)
                    current_sampled_value = torch.normal(mu_t1_batch, sigma_t1_batch)

                else:
                    alpha_t1 = F.softplus(params_t_plus_1[0]) + 1e-6
                    beta_t1 = F.softplus(params_t_plus_1[1]) + 1e-6
                    alpha_t1_batch = alpha_t1.repeat(num_scenarios)
                    beta_t1_batch = beta_t1.repeat(num_scenarios)
                    current_sampled_value = torch.distributions.Beta(alpha_t1_batch, beta_t1_batch).sample()

                # Shape [24, 2] (24 steps of features, starting from t+1)
                # Note: future_features must include features for t+1 through t+24
                # Since your loop starts at t+2, we extract 24 steps and index accordingly.
                future_features = self.data[day + start_forecast: day + end_forecast, -1, 1:].to(self.device)

                # h_initial_single shape: [num_layers, 1, hidden_size]
                # Replicate the initial state num_scenarios times along the batch dimension (dim=1)
                h_s = h_initial_single.unsqueeze(1).repeat(1, num_scenarios, 1)  # Shape: [num_layers, num_scenarios, hidden_size]
                c_s = c_initial_single.unsqueeze(1).repeat(1, num_scenarios, 1)  # Shape: [num_layers, num_scenarios, hidden_size]

                # 2. Sample the First Value Y_{t+1} for the entire batch
                # Sample the entire batch of num_scenarios values for Y_{t+1}
                # Shape: [num_scenarios]

                # Initialize the scenario path tensor [24 steps, num_scenarios]
                scenario_path_tensor = torch.empty(self.HOURS_IN_DAY, num_scenarios, device=self.device)
                scenario_path_tensor[0] = current_sampled_value  # Store Y_{t+1}

                # rest of the hours
                for i in range(start_forecast, end_forecast):  # Loop for 23 more steps (k=1 to 23)

                    # 1. Prepare the single-step input vector
                    # current_sampled_value (Y_{t+i-1}) shape: [num_scenarios]

                    # current_feature (Hour_{t+i}, Month_{t+i}) shape: [2]
                    current_feature = future_features[i - 1]  # Uses features for t+i

                    # Combine [Y_{t+i-1} (num_scenarios), Hour (2), Month (2)] -> shape [num_scenarios, 3]
                    # Must replicate the exogenous features num_scenarios times
                    features_batch = current_feature.unsqueeze(0).repeat(num_scenarios, 1)
                    input_vector_batch = torch.cat(
                        (current_sampled_value.unsqueeze(1), features_batch), dim=1
                    )  # Shape: [num_scenarios, 3]

                    # LSTM Input shape: [seq_len=1, batch_size=num_scenarios, input_size=3]
                    lstm_input = input_vector_batch.unsqueeze(1)

                    # 2. Run the LSTM on the entire batch
                    # out shape: [1, num_scenarios, hidden_size]
                    out, (h_s, c_s) = self.model.lstm(lstm_input, (h_s, c_s))
                    # 3. Calculate parameters and sample the new value
                    # out.squeeze(0) shape: [num_scenarios, hidden_size]
                    params = self.model.linear(out)  # params shape: [num_scenarios, 2]

                    if self.model_distribution == 'gaussian':
                        mu = params[:, 0, 0]  # shape: [num_scenarios]
                        sigma = F.softplus(params[:, 0, 1]) + 1e-6  # shape: [num_scenarios]
                        # Sample the next value Y_{t+i} for the entire batch
                        current_sampled_value = torch.normal(mu, sigma)

                    else:
                        alpha = F.softplus(params[:, 0, 0]) + 1e-6
                        beta = F.softplus(params[:, 0, 1]) + 1e-6
                        current_sampled_value = torch.distributions.Beta(alpha, beta).sample()

                    # Store the result
                    scenario_path_tensor[i] = current_sampled_value

                    # 4. Store the completed 24-step scenario paths
                    # Shape: [num_scenarios, 24 steps] (transposed)
                scenario_data[day_nr, :, :] = scenario_path_tensor

                if verbose > 0:
                    print(str(num_scenarios) + " scenarios generated for day " + str(day_nr))

        return scenario_data

    def reduce_scenarios(self, target_x, method='random'):
        """
        tensor_data: (days, 24, num_scenarios)
        target_x: Number of scenarios to keep
        """
        # Reshape to (days, scenarios, 24) for easier indexing by scenario
        # We want to reduce 'scenarios' for EVERY 'day'
        data = self.data.transpose(0, 2, 1)
        days, num_scens, hours = data.shape

        reduced_data = []

        for d in range(days):
            day_scenarios = data[d]  # Shape (num_scens, 24)

            if method == 'kmedoids':
                # K-Medoids picks actual scenarios as cluster centers
                kmed = KMedoids(n_clusters=target_x, method='pam', init='k-medoids++')
                kmed.fit(day_scenarios)
                reduced = kmed.cluster_centers_

            else:
                reduced = _fast_forward_selection(day_scenarios, target_x)

            reduced_data.append(reduced)

        # Return as (days, 24, target_x)
        return np.array(reduced_data).transpose(0, 2, 1)

def _fast_forward_selection(scenarios, n_target):
    """
    Heuristic to minimize Kantorovich distance.
    Complexity: O(n_target * num_scenarios)
    """
    n_total = scenarios.shape[0]
    # Calculate all-to-all Euclidean distance matrix
    # dists[i, j] is the distance between scenario i and j
    from scipy.spatial.distance import cdist
    dists = cdist(scenarios, scenarios, metric='euclidean')

    selected_indices = []
    remaining_indices = list(range(n_total))

    # 1. Start with the scenario that has the minimum distance to all others
    first_idx = np.argmin(np.sum(dists, axis=0))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # 2. Iteratively add scenarios
    while len(selected_indices) < n_target:
        # For each remaining scenario, calculate how much it would reduce
        # the distance if added to the selected set.
        min_dists = np.min(dists[selected_indices][:, remaining_indices], axis=0)
        best_new_idx_in_remaining = np.argmax(min_dists)

        best_idx = remaining_indices[best_new_idx_in_remaining]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return scenarios[selected_indices]