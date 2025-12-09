import torch
import numpy as np
import properscoring as ps
# No need for torch.distributions here, as we use the generated samples directly.

def calculate_crps_ensemble(samples: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Continuous Ranked Probability Score (CRPS)
    using the sample-based (ensemble) approximation.

    This is the required method for evaluating multi-step autoregressive
    forecasts (like DeepAR) where the distribution parameters (mu, sigma)
    are scenario-dependent (i.e., you only have samples).

    Args:
        samples (torch.Tensor): Tensor of predicted samples.
                                Shape MUST be [N_forecasts, N_scenarios]
                                where N_forecasts is the number of (day * time_step) pairs.
        obs (torch.Tensor): Tensor of observed true values (y).
                            Shape MUST be [N_forecasts].

    Returns:
        torch.Tensor: Mean CRPS over all observations.
    """
    # Ensure inputs are NumPy arrays for properscoring
    samples_np = samples.detach().cpu().numpy()
    obs_np = obs.detach().cpu().numpy()

    # Use crps_ensemble (sample-based CRPS)
    # obs_np: [N_forecasts]
    # samples_np: [N_forecasts, N_scenarios]
    crps_scores = ps.crps_ensemble(obs_np, samples_np)

    # Convert the mean score back to a PyTorch tensor
    mean_crps = torch.tensor(np.mean(crps_scores), dtype=samples.dtype)

    return mean_crps