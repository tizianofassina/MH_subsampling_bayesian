import numpy as np
import scipy.stats as stats
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from MH_algorithm_not_working import Kernel, C_data, MH_bayesian, MH_bayesian_subsampling


class NormalParamsKernel(Kernel):
    def __init__(self, sigma_mu, sigma_std):
        """
        Mixed kernel for normal distribution parameters:
        - Random walk in real space for mean
        - Random walk in log space for standard deviation
        """
        super().__init__(sigma_mu=sigma_mu, sigma_std=sigma_std)
        self.sigma_mu = sigma_mu
        self.sigma_std = sigma_std

    def sample(self, theta):
        """Generate new sample."""
        # theta[:, 0] is the mean, theta[:, 1] is the standard deviation
        new_theta = theta.copy()

        # Random walk for mean
        new_theta[:, 0] = theta[:, 0] + stats.norm().rvs(size=theta.shape[0]) * self.sigma_mu

        # Log-random walk for std dev
        log_std = np.log(theta[:, 1])
        new_log_std = log_std + stats.norm().rvs(size=theta.shape[0]) * self.sigma_std
        new_theta[:, 1] = np.exp(new_log_std)

        return new_theta

    def density(self, theta_proposition, theta_k):
        """Compute the density of the proposal."""
        # Density for mean component (normal)
        density_mu = stats.norm(theta_k[:, 0], self.sigma_mu).pdf(theta_proposition[:, 0])

        # Density for std dev component (log-normal)
        log_std_k = np.log(theta_k[:, 1])
        log_std_proposition = np.log(theta_proposition[:, 1])
        # Include Jacobian term for the log transformation
        density_std = stats.norm(log_std_k, self.sigma_std).pdf(log_std_proposition) / theta_proposition[:, 1]

        return density_mu * density_std


def create_mcmc_interactive_plots(MH_result, MH_subsample_result, mh_mu_values, mh_sigma_values, 
                               sub_mu_values, sub_sigma_values, samples_used, true_mu, true_sigma):
    """
    Creates interactive Plotly visualizations to compare MCMC results
    
    Parameters:
    - MH_result: Samples from standard MH
    - MH_subsample_result: Samples from subsampling MH
    - mh_mu_values: Mean parameter values from standard MH
    - mh_sigma_values: Std dev parameter values from standard MH
    - sub_mu_values: Mean parameter values from subsampling MH
    - sub_sigma_values: Std dev parameter values from subsampling MH
    - samples_used: Percentage of samples used by subsampling MH
    - true_mu: True mean parameter value
    - true_sigma: True std dev parameter value
    
    Returns:
    - Plotly figure
    """
    # Calculate error values
    mh_mu_errors = [np.abs(mu - true_mu) for mu in mh_mu_values]
    mh_sigma_errors = [np.abs(sigma - true_sigma) for sigma in mh_sigma_values]
    sub_mu_errors = [np.abs(mu - true_mu) for mu in sub_mu_values]
    sub_sigma_errors = [np.abs(sigma - true_sigma) for sigma in sub_sigma_values]
    
    # Create a dashboard with 6 subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Evolution of Mean Parameter (μ)', 'Evolution of Standard Deviation Parameter (σ)',
                       'Evolution of Error for μ', 'Evolution of Error for σ',
                       'Joint Distribution of Parameters (Last 50 Iterations)', 'Efficiency of Subsampling'),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Evolution of μ
    fig.add_trace(
        go.Scatter(x=list(range(len(mh_mu_values))), y=mh_mu_values, 
                   mode='lines', name='MH Standard', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(sub_mu_values))), y=sub_mu_values, 
                   mode='lines', name='MH Subsampling', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, len(mh_mu_values)], y=[true_mu, true_mu], 
                   mode='lines', name='True Value', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # 2. Evolution of σ
    fig.add_trace(
        go.Scatter(x=list(range(len(mh_sigma_values))), y=mh_sigma_values, 
                   mode='lines', name='MH Standard', line=dict(color='blue'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(sub_sigma_values))), y=sub_sigma_values, 
                   mode='lines', name='MH Subsampling', line=dict(color='orange'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, len(mh_sigma_values)], y=[true_sigma, true_sigma], 
                   mode='lines', name='True Value', line=dict(color='red', dash='dash'), showlegend=False),
        row=1, col=2
    )
    
    # 3. Error evolution for μ
    fig.add_trace(
        go.Scatter(x=list(range(len(mh_mu_errors))), y=mh_mu_errors, 
                   mode='lines', name='MH Standard', line=dict(color='blue'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(sub_mu_errors))), y=sub_mu_errors, 
                   mode='lines', name='MH Subsampling', line=dict(color='orange'), showlegend=False),
        row=2, col=1
    )
    
    # 4. Error evolution for σ
    fig.add_trace(
        go.Scatter(x=list(range(len(mh_sigma_errors))), y=mh_sigma_errors, 
                   mode='lines', name='MH Standard', line=dict(color='blue'), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(sub_sigma_errors))), y=sub_sigma_errors, 
                   mode='lines', name='MH Subsampling', line=dict(color='orange'), showlegend=False),
        row=2, col=2
    )
    
    # 5. Scatter plot of final samples
    standard_samples = MH_result[-50:].reshape(-1, 2)
    subsampling_samples = MH_subsample_result[-50:].reshape(-1, 2)
    
    fig.add_trace(
        go.Scatter(x=standard_samples[:, 0], y=standard_samples[:, 1], 
                   mode='markers', name='MH Standard', marker=dict(color='blue', size=8, opacity=0.5)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=subsampling_samples[:, 0], y=subsampling_samples[:, 1], 
                   mode='markers', name='MH Subsampling', marker=dict(color='orange', size=8, opacity=0.5)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=[true_mu], y=[true_sigma], 
                   mode='markers', name='True Value', 
                   marker=dict(color='red', size=15, symbol='star')),
        row=3, col=1
    )
    
    # 6. Percentage of samples used
    fig.add_trace(
        go.Scatter(x=list(range(len(samples_used))), y=samples_used, 
                   mode='lines', name='Samples Used (%)', line=dict(color='green')),
        row=3, col=2
    )
    
    # Update axes and layout
    fig.update_yaxes(title_text='μ', row=1, col=1)
    fig.update_yaxes(title_text='σ', row=1, col=2)
    fig.update_yaxes(title_text='Absolute Error', row=2, col=1, type='log')
    fig.update_yaxes(title_text='Absolute Error', row=2, col=2, type='log')
    fig.update_yaxes(title_text='σ', row=3, col=1)
    fig.update_yaxes(title_text='Percentage of Samples Used', row=3, col=2)
    
    fig.update_xaxes(title_text='Iteration', row=1, col=1)
    fig.update_xaxes(title_text='Iteration', row=1, col=2)
    fig.update_xaxes(title_text='Iteration', row=2, col=1)
    fig.update_xaxes(title_text='Iteration', row=2, col=2)
    fig.update_xaxes(title_text='μ', row=3, col=1)
    fig.update_xaxes(title_text='Iteration', row=3, col=2)
    
    fig.update_layout(
        title_text="MCMC With Subsampling - Performance Comparison",
        height=1200,
        width=1200
    )
    
    return fig

def create_convergence_speed_comparison(mh_mu_errors, mh_sigma_errors, sub_mu_errors, sub_sigma_errors, mh_time, sub_time):
    """
    Creates a plot comparing convergence speed with respect to wall clock time
    
    Parameters:
    - mh_mu_errors: Mean parameter errors from standard MH
    - mh_sigma_errors: Std dev parameter errors from standard MH
    - sub_mu_errors: Mean parameter errors from subsampling MH
    - sub_sigma_errors: Std dev parameter errors from subsampling MH
    - mh_time: Total time for standard MH
    - sub_time: Total time for subsampling MH
    
    Returns:
    - Plotly figure
    """
    fig_convergence = go.Figure()
    
    iteration_times_mh = np.linspace(0, mh_time, len(mh_mu_errors))
    iteration_times_sub = np.linspace(0, sub_time, len(sub_mu_errors))
    
    fig_convergence.add_trace(
        go.Scatter(x=iteration_times_mh, y=mh_mu_errors, 
                  mode='lines', name='MH Standard - μ Error', line=dict(color='blue'))
    )
    fig_convergence.add_trace(
        go.Scatter(x=iteration_times_mh, y=mh_sigma_errors, 
                  mode='lines', name='MH Standard - σ Error', line=dict(color='blue', dash='dash'))
    )
    fig_convergence.add_trace(
        go.Scatter(x=iteration_times_sub, y=sub_mu_errors, 
                  mode='lines', name='MH Subsampling - μ Error', line=dict(color='orange'))
    )
    fig_convergence.add_trace(
        go.Scatter(x=iteration_times_sub, y=sub_sigma_errors, 
                  mode='lines', name='MH Subsampling - σ Error', line=dict(color='orange', dash='dash'))
    )
    
    fig_convergence.update_layout(
        title='Convergence Speed Comparison',
        xaxis_title='Time (seconds)',
        yaxis_title='Absolute Error',
        yaxis_type='log',
        height=600,
        width=1000
    )
    
    return fig_convergence

def run_mcmc_comparison(n=100000, sample_length=40, n_iter=600):
    """
    Function executing the MCMC comparison between standard MH and subsampling MH
    
    Parameters:
    - n: Number of data points
    - sample_length: Number of parallel chains
    - n_iter: Number of iterations
    
    Returns:
    - data: Generated data
    - samples_used: Percentage of samples used by subsampling MH
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate test data from normal distribution
    true_mu = 5.0
    true_sigma = 2.0
    data = stats.norm(true_mu, true_sigma).rvs(n)

    # Define a moderately informative prior
    def prior_density(theta):
        # Normal prior for mean, inverse gamma for variance
        prior_mu = stats.norm(0, 10).pdf(theta[:, 0])  # Vague normal prior for mean
        prior_sigma = stats.invgamma(2, scale=3).pdf(theta[:, 1])  # Inverse gamma for std dev
        return prior_mu * prior_sigma

    # Define the likelihood function for normal model
    def likelihood(x, theta):
        mu = theta[:, 0][np.newaxis, :]
        sigma = theta[:, 1][np.newaxis, :]
        return stats.norm(mu, sigma).pdf(x[:, np.newaxis])

    # Parameters for the algorithm
    sigma_mu = 0.2  # Step size for mean
    sigma_std = 0.1  # Step size for std dev in log space
    kernel = NormalParamsKernel(sigma_mu, sigma_std)

    # Starting point for chains - closer to true values to help convergence
    theta_0 = np.zeros((sample_length, 2))
    theta_0[:, 0] = 3.0  # Initial guess for mean
    theta_0[:, 1] = 1.5  # Initial guess for std dev

    # Subsampling parameters
    delta_base = 0.01  # Base error probability
    max_t_look = 100
    delta_t = delta_base * np.power(0.95, np.arange(max_t_look))
    gamma = 2.0  # Growth factor for batch size (as in the paper)

    # Create the C function
    C = C_data(data, likelihood)

    # Run standard MH algorithm
    print("Running standard MH algorithm...")
    start_time_mh = time.time()
    MH_result, mh_mu_values, mh_sigma_values = MH_bayesian(
        sample_length=sample_length,
        n_iter=n_iter,
        kernel=kernel,
        prior_density=prior_density,
        likelihood=likelihood,
        data=data,
        theta_0=theta_0
    )
    mh_time = time.time() - start_time_mh
    print(f"MH Standard - Total time: {mh_time:.2f}s")

    # Run MH with subsampling
    print("\nRunning MH with subsampling...")
    start_time_sub = time.time()
    MH_subsample_result, samples_used, sub_mu_values, sub_sigma_values = MH_bayesian_subsampling(
        sample_length=sample_length,
        gamma=gamma,
        C_fn=C,
        n_iter=n_iter,
        kernel=kernel,
        prior_density=prior_density,
        likelihood=likelihood,
        data=data,
        theta_0=theta_0,
        delta=delta_t
    )
    sub_time = time.time() - start_time_sub
    print(f"MH Subsampling - Total time: {sub_time:.2f}s")

    # Display results
    print("\nFinal results:")
    print(f"True values: mu = {true_mu}, sigma = {true_sigma}")
    print(f"Standard MH: mu = {np.mean(MH_result[-1, :, 0]):.4f}, sigma = {np.mean(MH_result[-1, :, 1]):.4f}")
    print(f"Subsampling MH: mu = {np.mean(MH_subsample_result[-1, :, 0]):.4f}, sigma = {np.mean(MH_subsample_result[-1, :, 1]):.4f}")
    print(f"\nExecution time:")
    print(f"Speed gain: {mh_time/sub_time:.2f}x")

    # Calculate errors for convergence plots
    mh_mu_errors = [np.abs(mu - true_mu) for mu in mh_mu_values]
    mh_sigma_errors = [np.abs(sigma - true_sigma) for sigma in mh_sigma_values]
    sub_mu_errors = [np.abs(mu - true_mu) for mu in sub_mu_values]
    sub_sigma_errors = [np.abs(sigma - true_sigma) for sigma in sub_sigma_values]

    # Create interactive visualizations
    mcmc_fig = create_mcmc_interactive_plots(
        MH_result, MH_subsample_result, 
        mh_mu_values, mh_sigma_values, 
        sub_mu_values, sub_sigma_values, 
        samples_used, true_mu, true_sigma
    )
    
    mcmc_fig.show()
    
    # Create convergence speed comparison
    fig_convergence = create_convergence_speed_comparison(
        mh_mu_errors, mh_sigma_errors, 
        sub_mu_errors, sub_sigma_errors, 
        mh_time, sub_time
    )
    
    fig_convergence.show()

    return data, samples_used

if __name__ == "__main__":
    run_mcmc_comparison()
