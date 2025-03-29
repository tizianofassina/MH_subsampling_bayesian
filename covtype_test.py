import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from MH_algorithm import MH_bayesian, MH_bayesian_subsampling, C_data

# ---------------------------
# Define the proposal kernel
# ---------------------------
class Kernel:
    def __init__(self, **params):
        """
        General stateful proposal kernel.
        Parameters:
        - **params: Dictionary of kernel parameters.
        """
        self.params = params

    def sample(self, theta_k: np.ndarray) -> np.ndarray:
        """Generate a new sample given theta (to be implemented by subclasses)."""
        raise NotImplementedError

    def density(self, theta_proposition: np.ndarray, theta_k: np.ndarray) -> np.ndarray:
        """Compute the density p(theta_proposition | theta_k) (to be implemented by subclasses)."""
        raise NotImplementedError

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        super().__init__(sigma=sigma)
        self.sigma = sigma

    def sample(self, theta_k: np.ndarray) -> np.ndarray:
        # Propose a new state by adding Gaussian noise
        return theta_k + np.random.normal(0, self.sigma, size=theta_k.shape)

    def density(self, theta_proposition: np.ndarray, theta_k: np.ndarray) -> np.ndarray:
        # Compute the density for each chain assuming independent Gaussian noise on each coordinate.
        dens = stats.norm.pdf(theta_proposition, loc=theta_k, scale=self.sigma)
        return np.prod(dens, axis=1)

# ---------------------------
# Define likelihood and prior
# ---------------------------
def likelihood(data, theta: np.ndarray) -> np.ndarray:
    """
    Compute the logistic regression likelihood.
    data is a tuple (X, y) with:
      - X: design matrix of shape (n, d)
      - y: binary labels of shape (n,)
    theta is an array of shape (m, d), where m is the number of chains.
    Returns a (n, m) array of likelihood values.
    """
    #print(type(data))
    X, y = data[:, 0:-1], data[:, -1]
    # Compute the linear predictors: shape (n, m)
    logits = X.dot(theta.T)
    # Sigmoid function for probabilities
    probs = 1 / (1 + np.exp(-logits))
    # Compute likelihood for each data point and each chain:
    # p(y|x,θ) = p^y * (1-p)^(1-y)
    y = y[:, np.newaxis]  # reshape to (n,1) for broadcasting
    L = (probs ** y) * ((1 - probs) ** (1 - y))
    return L

def cauchy_prior(theta: np.ndarray, loc=0, scale=1) -> np.ndarray:
    """
    Compute the independent Cauchy prior density for each chain.
    theta is of shape (m, d). For each parameter component:
      f(x) = 1 / (π * scale * [1 + ((x - loc)/scale)^2])
    The joint density is the product over the d coordinates.
    Returns an array of shape (m,).
    """
    pdf_vals = stats.cauchy.pdf(theta, loc=loc, scale=scale)
    return np.prod(pdf_vals, axis=1)

# ---------------------------
# Error computation
# ---------------------------
def compute_error(X, y, theta):
    """
    Compute the classification error (misclassification rate) given:
      - X: feature matrix of shape (n, d)
      - y: true binary labels of shape (n,)
      - theta: parameter vector of shape (m, d)
    Uses a threshold of 0.5 on the sigmoid probability.
    """
    logits = X.dot(theta.T)
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(int)
    error = np.mean(predictions != y, axis=0)
    return error

def moving_average(data, window_size):
    """Compute the moving average of a 1D array with boundary adjustments.
    
    The moving average at each index i is computed over the window:
    [max(0, i - (window_size-1)//2) : min(len(data), i + window_size//2 + 1)]
    so that near the edges the average is taken over a smaller set of values.
    """
    n = len(data)
    result = np.empty(n)
    # Compute the cumulative sum with a zero prepended
    cumsum = np.cumsum(np.insert(data, 0, 0))
    half1 = (window_size - 1) // 2
    half2 = window_size // 2
    for i in range(n):
        start = max(0, i - half1)
        end = min(n, i + half2 + 1)
        window_sum = cumsum[end] - cumsum[start]
        result[i] = window_sum / (end - start)
    return result


# ---------------------------
# Main experiment
# ---------------------------
def main(q=2):
    # Load covtype dataset (from sklearn)
    # Note: The original dataset has 581,012 samples and 54 features.
    # Here we select the first 400,000 samples and only the first 2 quantitative features.
    print("Loading the covtype dataset...")
    data_cov = fetch_covtype()
    X_all = data_cov.data.astype(np.float64)
    y = data_cov.target
    # Convert the multiclass target into a binary classification problem.
    # Here we set y=1 if the target equals 2 and 0 otherwise.
    y_binary_all = (y == 2).astype(int)
    
    n_samples = 400000
    X = X_all[:n_samples, :q]  # select the first 2 attributes
    y_binary = y_binary_all[:n_samples]
    X_test = X_all[n_samples:, :q]
    y_test = y_binary_all[n_samples:]
    
    # Preprocess the features (standardization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Package data as a tuple (X, y)
    data = np.concatenate((X, y_binary[:, np.newaxis]), axis=1)
    
    # Set MH sampler parameters
    sample_length = 4  # number of chains (starting points)
    gamma = 2

    C_function = C_data(data, likelihood)
    delta_base = 0.01  # Base error probability
    max_t_look = 1000
    delta_t = delta_base * np.power(0.95, np.arange(max_t_look))
    n_iter = 300       # number of iterations (set small for testing; increase as needed)
    sigma = 0.1        # proposal standard deviation (tuning parameter)
    
    # Initialize the proposal kernel
    kernel = GaussianKernel(sigma=sigma)
    
    # Draw four random starting points for the chains (each has dimension equal to number of features, here d=2)
    d = X.shape[1]
    theta_0 = np.random.randn(sample_length, d)
    
    # Run the Metropolis-Hastings sampler
   
    print("Running MH subsampling sampler...")
    start = time.time()
    chain_sub, per_data = MH_bayesian_subsampling(sample_length, gamma, C_function, n_iter, kernel, cauchy_prior, likelihood, data, theta_0, delta_t, percentage=True)
    end_sub = time.time() - start
    print("Time MH Subsampling: ", end_sub)
    start = time.time()
    chain = MH_bayesian(sample_length, n_iter, kernel, cauchy_prior, likelihood, data, theta_0)
    end = time.time() - start
    print("Time MH: ", end)
    print("MH subsampling was ", end / end_sub, " times faster.")
    
    # Plot the trajectories for each chain in the 2D parameter space.
    plt.figure(figsize=(10, 5))
    for i in range(sample_length):
        plt.plot(chain[:, i, 0], chain[:, i, 1], marker='o', label=f'Chain {i+1}')
    plt.xlabel('Theta[0]')
    plt.ylabel('Theta[1]')
    plt.title('MH Sampler Trajectories on Covtype Experiment (2D parameter space)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(sample_length):
        plt.plot(chain_sub[:, i, 0], chain_sub[:, i, 1], marker='o', label=f'Chain {i+1}')
    plt.xlabel('Theta[0]')
    plt.ylabel('Theta[1]')
    plt.title('MH Subsampling Trajectories on Covtype Experiment (2D parameter space)')
    plt.legend()
    plt.savefig('chain_covtype_experiment')
    plt.show()
    
    # ---------------------------
    # Compute running statistics for each chain
    # ---------------------------
    n_steps = chain.shape[0]
    mean_per = np.mean(per_data, axis=0)

    if q == 2:
        running_mean = np.zeros_like(chain)        # shape: (n_steps, sample_length, d)
        running_quantile = np.zeros_like(chain)      # shape: (n_steps, sample_length, d)
        running_mean_sub = np.zeros_like(chain_sub)
        running_quantile_sub = np.zeros_like(chain_sub)
        
        for i in range(sample_length):
            for t in range(n_steps):
                running_mean[t, i, :] = np.mean(chain[:t+1, i, :], axis=0)
                running_quantile[t, i, :] = np.quantile(chain[:t+1, i, :], 0.3, axis=0)
                running_mean_sub[t, i, :] = np.mean(chain_sub[:t+1, i, :], axis=0)
                running_quantile_sub[t, i, :] = np.quantile(chain_sub[:t+1, i, :], 0.3, axis=0)
        
        # ---------------------------
        # Figure 1: theta[0] statistics
        # Two subplots: left for running means, right for running 30% quantiles (across the 4 chains)
        # ---------------------------
        fig0, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['blue', 'green', 'red', 'pink']
        for i in range(sample_length):
            ax0.plot(range(n_steps), running_mean[:, i, 0], color=colors[i])
            ax1.plot(range(n_steps), running_quantile[:, i, 0], color=colors[i])
            ax0.plot(range(n_steps), running_mean_sub[:, i, 0], label=f'{int(100 * mean_per[i])}% of n', linestyle='dashed', color=colors[i])
            ax1.plot(range(n_steps), running_quantile_sub[:, i, 0], label=f'{int(100 * mean_per[i])}% of n', linestyle='dashed', color=colors[i])

            ax2.plot(range(n_steps), running_mean[:, i, 1], color = colors[i])
            ax3.plot(range(n_steps), running_quantile[:, i, 1], color = colors[i])
            ax2.plot(range(n_steps), running_mean_sub[:, i, 1], label=f'{int(100 * mean_per[i])}% of n', linestyle='dashed', color = colors[i])
            ax3.plot(range(n_steps), running_quantile_sub[:, i, 1], label=f'{int(100 * mean_per[i])}% of n', linestyle='dashed', color=colors[i])

        ax0.set_title('Running Mean for θ[0]')
        ax0.set_xlabel('Iteration')
        ax0.set_ylabel('Mean')
        ax1.set_title('Running 30% Quantile for θ[0]')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('30% Quantile')
        ax2.set_title('Running Mean for θ[1]')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean')
        ax3.set_title('Running 30% Quantile for θ[1]')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('30% Quantile')
        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.tight_layout()
        plt.savefig('stats_covtype_experiment')
        plt.show()

    # ---------------------------
    # Compute train and test errors over iterations.
    # ---------------------------
    train_errors = np.zeros((n_steps, sample_length))
    test_errors = np.zeros((n_steps, sample_length))
    train_errors_sub = np.zeros((n_steps, sample_length))
    test_errors_sub = np.zeros((n_steps, sample_length))
    for t in range(chain.shape[0]):
        train_errors[t] = compute_error(X, y_binary[:, np.newaxis], chain[t])
        test_errors[t] = compute_error(X_test, y_test[:, np.newaxis], chain[t])

        train_errors_sub[t] = compute_error(X, y_binary[:, np.newaxis], chain_sub[t])
        test_errors_sub[t] = compute_error(X_test, y_test[:, np.newaxis], chain_sub[t])


    # Plot the evolution of training and test errors over iterations.
    fig1, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(sample_length):
        train_i = moving_average(train_errors[:, i],window_size=20)
        ax0.plot(range(n_steps), train_i)

        test_i = moving_average(test_errors[:, i], window_size=20)
        ax1.plot(range(n_steps), test_i)

        train_i = moving_average(train_errors_sub[:, i], window_size=20)
        ax0.plot(range(n_steps), train_i, label=f'{int(100 * mean_per[i])}% of n', linestyle='dashed')

        test_i = moving_average(test_errors_sub[:, i],window_size=20)
        ax1.plot(range(n_steps), test_i, label=f'{int(100 * mean_per[i])}% of n', linestyle='dashed')

    ax0.set_title('Mean training error')
    ax0.set_xlabel('Iteration')
    ax0.set_ylabel('Misclassification error')
    ax1.set_title('Mean test error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Misclassification error')
    ax0.legend()
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f'errors_covtype_experiment_q{q}')
    plt.show()



if __name__ == '__main__':
    q=10
    main(q=q)

