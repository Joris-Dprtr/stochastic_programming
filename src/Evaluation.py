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

def masked_interval_coverage(samples, y_true, alpha=0.8, eps=1e-5):
    """
    Computes overall empirical coverage, excluding zero-production periods.

    samples: (D, H, S)
    y_true:  (D, H)
    alpha: nominal interval level
    eps: threshold to define daylight / non-zero PV
    """
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q

    # Quantiles over scenarios
    lower = np.quantile(samples, lower_q, axis=-1)
    upper = np.quantile(samples, upper_q, axis=-1)

    # Daylight mask
    mask = y_true > eps

    # Coverage indicator
    covered = (y_true >= lower) & (y_true <= upper)

    # Aggregate only valid points
    if mask.sum() == 0:
        return np.nan

    return covered[mask].mean()