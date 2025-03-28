import numpy as np
import scipy.stats as stats
import math
from typing import Callable

class Kernel:
    """Base class for MCMC proposal kernels"""
    def __init__(self, **params):
        self.params = params
        
    def sample(self, theta_k: np.ndarray) -> np.ndarray:
        """Generate a new sample given theta"""
        raise NotImplementedError
        
    def density(self, theta_proposition: np.ndarray, theta_k: np.ndarray) -> np.ndarray:
        """Compute the density p(theta_proposition | theta_k)"""
        raise NotImplementedError


def C_data(data, likelihood_function):
    """
    Create a function to compute the upper bound C for the log likelihood ratios.
    """
    def C(theta_1, theta_2):
        # For normal distribution, the log-likelihood difference can be bounded more efficiently
        # Based on the paper, C is related to max|θ1-θ2| * max|x|
        max_abs_x = np.max(np.abs(data))
        C_values = np.zeros(theta_1.shape[0])
        
        for i in range(theta_1.shape[0]):
            mu1, sigma1 = theta_1[i]
            mu2, sigma2 = theta_2[i]
            
            # Bound for the mean difference (based on normal log-likelihood)
            mu_diff = np.abs(mu1 - mu2)
            sigma_diff = np.abs(1/sigma1**2 - 1/sigma2**2)
            
            # This is a tighter bound specific to normal distributions
            bound1 = mu_diff * max_abs_x / min(sigma1, sigma2)**2
            bound2 = sigma_diff * max_abs_x**2 / 2
            bound3 = np.abs(np.log(sigma1/sigma2))
            
            C_values[i] = bound1 + bound2 + bound3
            
        return C_values
    
    return C

def MH_bayesian(sample_length, n_iter, kernel, prior_density, likelihood, data, theta_0, burnin=0.2):
    """
    Standard Metropolis-Hastings algorithm for Bayesian inference.
    
    Parameters:
    - sample_length: Number of parallel chains
    - n_iter: Number of iterations
    - kernel: Proposal kernel
    - prior_density: Prior density function
    - likelihood: Likelihood function
    - data: Data for inference
    - theta_0: Initial parameter values
    - burnin: Proportion of samples to discard as burn-in
    
    Returns:
    - MCMC samples and diagnostics
    """
    n_burnin = int(burnin * n_iter)
    theta_k = [theta_0]
    acceptance_rate = 0

    # Store mean values for plotting
    mean_values = []
    std_values = []

    for i in range(n_iter):
        u = stats.uniform(0, 1).rvs(size=sample_length)
        theta_proposition = kernel.sample(theta_k[-1])

        # Calculate the MH acceptance ratio (in log space)
        psi = (np.log(u) + np.log(np.maximum(prior_density(theta_k[-1]), 1e-10)) +
               np.log(np.maximum(kernel.density(theta_proposition, theta_k[-1]), 1e-10)) -
               np.log(np.maximum(prior_density(theta_proposition), 1e-10)) -
               np.log(np.maximum(kernel.density(theta_k[-1], theta_proposition), 1e-10))
              )

        # Compute log-likelihood ratio
        lambd = np.sum(np.log(np.maximum(likelihood(data, theta_proposition), 1e-10)) -
                       np.log(np.maximum(likelihood(data, theta_k[-1]), 1e-10)), axis=0)

        # Accept or reject
        accepted = lambd > psi

        acceptance_rate += np.mean(accepted) / n_iter


        # Update chain
        theta_new = theta_k[-1].copy()
        theta_new[accepted] = theta_proposition[accepted]
        theta_k.append(theta_new)
        
        # Store current means
        mean_values.append(np.mean(theta_new[:, 0]))
        std_values.append(np.mean(theta_new[:, 1]))

        # Print progress
        if (i+1) % 100 == 0:
            print(f"MH Standard - Iteration {i+1}/{n_iter}")

    print(f"MH Standard - Acceptance rate: {acceptance_rate:.4f}")
    
    # Return samples after burn-in and diagnostic information
    return np.array(theta_k[n_burnin:]), mean_values, std_values


def MH_bayesian_subsampling(sample_length, gamma, C_fn, n_iter, kernel, prior_density, 
                           likelihood, data, theta_0, delta, burnin=0.2):
    """
    Metropolis-Hastings algorithm with adaptive subsampling.
    
    Parameters:
    - sample_length: Number of parallel chains
    - gamma: Growth factor for batch size
    - C_fn: Function to compute concentration bounds
    - n_iter: Number of iterations
    - kernel: Proposal kernel
    - prior_density: Prior density function
    - likelihood: Likelihood function
    - data: Data for inference
    - theta_0: Initial parameter values
    - delta: Sequence of error probabilities
    - burnin: Proportion of samples to discard as burn-in
    
    Returns:
    - MCMC samples and diagnostics
    """
    n_burnin = int(burnin * n_iter)
    theta_k = [theta_0]
    samples_used = []
    acceptance_rate = 0
    
    # Store mean values for plotting
    mean_values = []
    std_values = []



    for i in range(n_iter):
        u = stats.uniform(0, 1).rvs(size=sample_length)
        theta_proposition = kernel.sample(theta_k[-1])
        n = np.shape(data)[0]

        # Calculate prior and proposal ratios
        log_prior_ratio = np.log(np.maximum(prior_density(theta_proposition), 1e-10)) - \
                         np.log(np.maximum(prior_density(theta_k[-1]), 1e-10))
        
        log_proposal_ratio = np.log(np.maximum(kernel.density(theta_k[-1], theta_proposition), 1e-10)) - \
                            np.log(np.maximum(kernel.density(theta_proposition, theta_k[-1]), 1e-10))
        
        psi = (np.log(u) + log_prior_ratio + log_proposal_ratio) / n

        # Initialize subsampling variables
        t = 0
        t_look = 0
        lambd = np.zeros(theta_0.shape[0])
        b = 1  # Start with batch size 1 (as in the paper)
        not_done = np.ones(theta_0.shape[0], dtype=bool)
        total_samples_this_iter = 0

        # Set up data tracking
        data_sampled = np.empty(0) if data.ndim == 1 else np.empty((0,) + data.shape[1:])
        data_to_be_sampled = data.copy()
        np.random.shuffle(data_to_be_sampled)  # Shuffle to sample without replacement

        # Main subsampling loop
        while np.any(not_done) and len(data_to_be_sampled) > 0:
            # Determine batch size for this iteration
            samples_needed = min(b - t, len(data_to_be_sampled))
            if samples_needed <= 0:
                break

            # Sample data batch
            new_samples = data_to_be_sampled[:samples_needed]
            data_to_be_sampled = data_to_be_sampled[samples_needed:]
            total_samples_this_iter += samples_needed
            data_sampled = np.concatenate((data_sampled, new_samples))

            # Calculate log-likelihood ratios for new batch
            current_batch_ll = np.log(np.maximum(likelihood(new_samples, theta_k[-1]), 1e-10))
            proposed_batch_ll = np.log(np.maximum(likelihood(new_samples, theta_proposition), 1e-10))
            log_ratios = proposed_batch_ll - current_batch_ll

            # Update running estimate of lambda (average log-likelihood ratio)
            if t > 0:
                # Only update for chains that haven't decided yet
                for j in range(theta_0.shape[0]):
                    if not_done[j]:
                        lambd[j] = (t * lambd[j] + np.sum(log_ratios[:, j])) / b
            else:
                # First batch
                for j in range(theta_0.shape[0]):
                    lambd[j] = np.sum(log_ratios[:, j]) / b

            # Get current delta value
            current_delta = delta[min(t_look, len(delta)-1)]
            
            # Calculate empirical standard deviation
            if len(data_sampled) > 1:
                all_log_ratios = np.zeros((len(data_sampled), theta_0.shape[0]))
                
                # Calculate all log-ratios to get proper empirical SD
                for j in range(0, len(data_sampled), 1000):  # Process in chunks to avoid memory issues
                    end_idx = min(j + 1000, len(data_sampled))
                    chunk = data_sampled[j:end_idx]
                    
                    current_chunk_ll = np.log(np.maximum(likelihood(chunk, theta_k[-1]), 1e-10))
                    proposed_chunk_ll = np.log(np.maximum(likelihood(chunk, theta_proposition), 1e-10))
                    all_log_ratios[j:end_idx] = proposed_chunk_ll - current_chunk_ll
                
                # Calculate empirical standard deviation for each chain
                empirical_sd = np.std(all_log_ratios, axis=0, ddof=1)
            else:
                # Conservative estimate if we don't have enough samples
                empirical_sd = np.ones(theta_0.shape[0])
            
            # Calculate C bounds for all chains
            c_values = C_fn(theta_proposition, theta_k[-1])
            
            # Compute concentration bounds using Bernstein inequality
            concentration_bounds = np.zeros(theta_0.shape[0])
            for j in range(theta_0.shape[0]):
                # Only compute for chains that haven't decided yet
                if not_done[j]:
                    concentration_bounds[j] = (np.sqrt(2 * empirical_sd[j]**2 * np.log(3/current_delta) / b) + 
                                             6 * c_values[j] * np.log(3/current_delta) / (3*b))
            
            # Update counters
            t = b
            t_look += 1
            b = min(n, math.ceil(gamma * b))  # Geometric growth of batch size
            
            # Update not_done status for each chain individually
            for j in range(theta_0.shape[0]):
                if not_done[j]:
                    if np.abs(lambd[j] - psi[j]) > concentration_bounds[j] or b > n:
                        not_done[j] = False

        # Record percentage of samples used
        samples_used.append(total_samples_this_iter / n * 100)

        # Make accept/reject decisions
        accepted = (lambd > psi)
        acceptance_rate += np.mean(accepted) / n_iter
        
        # Update chain
        theta_new = theta_k[-1].copy()
        theta_new[accepted] = theta_proposition[accepted]
        theta_k.append(theta_new)
        
        # Store current means
        mean_values.append(np.mean(theta_new[:, 0]))
        std_values.append(np.mean(theta_new[:, 1]))
        
        # Print progress
        if (i+1) % 100 == 0:
            print(f"MH Subsampling - Iteration {i+1}/{n_iter}, samples used: {total_samples_this_iter/n*100:.2f}%")

    print(f"MH Subsampling - Acceptance rate: {acceptance_rate:.4f}")
    print(f"Average percentage of samples used: {np.mean(samples_used):.2f}%")
    
    # Return samples after burn-in and diagnostic information
    return np.array(theta_k[n_burnin:]), samples_used, mean_values, std_values
