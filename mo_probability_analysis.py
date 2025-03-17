import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def proba(numbers, epsilon, t, num_samples):
    """
    Calculate the probability that the empirical mean is close to the true mean.
    
    Parameters:
    - numbers: data array
    - epsilon: proximity threshold
    - t: subsample size
    - num_samples: number of Monte Carlo samples
    
    Returns:
    - Probability for each epsilon
    """
    sampled_lines = np.array([np.random.choice(np.arange(numbers.shape[0]), size=t, replace=False) 
                              for _ in range(num_samples)])
    means = np.mean(numbers[sampled_lines], axis=1)
    A = np.mean(
        (np.abs(means[:, np.newaxis] - numbers.mean()) < epsilon[np.newaxis,:]),
        axis=0)
    return A


def proba_conditional(numbers, epsilon, t, num_samples, a):
    """
    Calculate the conditional probability that the empirical mean is close to the true mean
    given that it is far from reference value a.
    
    Parameters:
    - numbers: data array
    - epsilon: proximity threshold
    - t: subsample size
    - num_samples: number of Monte Carlo samples
    - a: reference values
    
    Returns:
    - Conditional probability
    """
    sampled_lines = np.array(
        [np.random.choice(np.arange(numbers.shape[0]), size=t, replace=False) for _ in range(num_samples)])

    means = np.mean(numbers[sampled_lines], axis=1)

    A = np.mean(
        (np.abs(means[:, np.newaxis, np.newaxis] - a[np.newaxis, :, np.newaxis]) > epsilon[np.newaxis, np.newaxis, :]) & 
        (np.abs(means[:, np.newaxis, np.newaxis] - numbers.mean()) < epsilon[np.newaxis, np.newaxis, :]),
        axis=0)

    B = np.mean(np.abs(means[:, np.newaxis, np.newaxis] - a[np.newaxis,:,np.newaxis]) > 
               epsilon[np.newaxis,np.newaxis, :], axis=0)

    return A / B


def create_interactive_probability_analysis(data=None):
    """
    Create interactive visualization of probability analysis
    
    Parameters:
    - data: data array (optional)
    
    Returns:
    - Plotly figure and computed values
    """
    # If no data provided, create test data
    if data is None:
        # Use exponential data
        np.random.seed(42)  # For reproducibility
        data = np.random.exponential(scale=2, size=200)
    
    # Parameters for analysis
    t = 15  # Subsample size
    mean = data.mean()
    print(f"Data mean: {mean:.4f}")
    print(f"Subsample size for probability analysis: {t}")
    
    # Define grids for a and epsilon
    a = np.arange(mean - 30, mean + 30, 1.0)  # Values around the mean
    epsilon = np.arange(0.2, 20., 0.5)  # Proximity thresholds
    
    print("Computing probabilities... (this may take a moment)")
    # Calculate probabilities
    proba_values = proba(numbers=data, epsilon=epsilon, t=t, num_samples=2000)
    proba_cond_values = proba_conditional(numbers=data, epsilon=epsilon, t=t, num_samples=2000, a=a)
    
    # Create grid for 3D plot
    proba_repeated = np.tile(proba_values, (a.shape[0], 1))
    X, Y = np.meshgrid(epsilon, a)
    
    # Create Plotly figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Simple Probability P(|μ̂ - μ| < ε)', 
                       'Conditional Probability P(|μ̂ - μ| < ε | |μ̂ - a| > ε)'),
        horizontal_spacing=0.05
    )
    
    # Surface for simple probability
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=proba_repeated,
            colorscale='Viridis',
            showscale=True,
            opacity=0.9,
            name="P(|μ̂ - μ| < ε)"
        ),
        row=1, col=1
    )
    
    # Surface for conditional probability
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=proba_cond_values,
            colorscale='Inferno',
            showscale=True,
            opacity=0.9,
            name="P(|μ̂ - μ| < ε | |μ̂ - a| > ε)"
        ),
        row=1, col=2
    )
    
    # Add line for true mean
    y_values = np.ones(100) * mean
    x_values = np.linspace(epsilon.min(), epsilon.max(), 100)
    z_values = np.zeros(100)
    
    fig.add_trace(
        go.Scatter3d(
            x=x_values, y=y_values, z=z_values,
            mode='lines',
            line=dict(color='red', width=5),
            name='True mean'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=x_values, y=y_values, z=z_values,
            mode='lines',
            line=dict(color='red', width=5),
            name='True mean'
        ),
        row=1, col=2
    )
    
    # Configure axes and labels
    fig.update_layout(
        title='Interactive comparison between simple and conditional probability',
        scene1=dict(
            xaxis_title='Epsilon (ε)',
            yaxis_title='Reference value (a)',
            zaxis_title='Probability',
            xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
            yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
            zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)')
        ),
        scene2=dict(
            xaxis_title='Epsilon (ε)',
            yaxis_title='Reference value (a)',
            zaxis_title='Probability',
            xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
            yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)'),
            zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)')
        ),
        height=800,
        width=1400,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    
    # Configure camera for easy exploration
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        scene2_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    return fig, proba_values, proba_cond_values, X, Y


def create_2d_probability_views(proba_values, proba_cond_values, epsilon, a):
    """
    Create complementary 2D visualizations for probabilities
    
    Parameters:
    - proba_values: Simple probability values
    - proba_cond_values: Conditional probability values
    - epsilon: Proximity threshold values
    - a: Reference values
    
    Returns:
    - Plotly figure
    """
    # Check dimensions to avoid index errors
    n_rows = proba_cond_values.shape[0]
    
    # Select a few a values for 2D plot
    a_indices = [0, n_rows//4, n_rows//2, 3*n_rows//4, n_rows-1]
    a_selected = [a[min(i, len(a)-1)] for i in a_indices]
    
    # Create 2D plot
    fig_2d = go.Figure()
    
    # Plot curves for different a values
    for i, a_val in enumerate(a_selected):
        idx = a_indices[i]
        if idx < n_rows:  # Check that index is within bounds
            fig_2d.add_trace(
                go.Scatter(
                    x=epsilon, 
                    y=proba_cond_values[idx, :],
                    mode='lines',
                    name=f'a = {a_val:.1f}'
                )
            )
    
    # Also plot simple probability for comparison
    fig_2d.add_trace(
        go.Scatter(
            x=epsilon,
            y=proba_values,
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='Simple probability'
        )
    )
    
    # Configure 2D plot
    fig_2d.update_layout(
        title='Conditional probability for different values of a',
        xaxis_title='Epsilon (ε)',
        yaxis_title='Probability',
        legend_title='Value of a',
        height=500,
        width=1000
    )
    
    return fig_2d


def create_probability_heatmap(X, Y, proba_cond_values, mean):
    """
    Create heatmap of conditional probabilities
    
    Parameters:
    - X, Y: Meshgrid for heatmap
    - proba_cond_values: Conditional probability values
    - mean: True mean of the data
    
    Returns:
    - Plotly figure
    """
    fig_heatmap = go.Figure(data=
        go.Heatmap(
            z=proba_cond_values,
            x=X[0, :],  # Epsilon values
            y=Y[:, 0],  # a values
            colorscale='Viridis',
            colorbar=dict(title='Conditional probability')
        )
    )
    
    # Add line for true mean
    fig_heatmap.add_shape(
        type="line",
        x0=X[0, 0],
        y0=mean,
        x1=X[0, -1],
        y1=mean,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Configure heatmap
    fig_heatmap.update_layout(
        title='Heatmap of conditional probability P(|μ̂ - μ| < ε | |μ̂ - a| > ε)',
        xaxis_title='Epsilon (ε)',
        yaxis_title='Reference value (a)',
        height=600,
        width=1000
    )
    
    return fig_heatmap


def analyze_subsampling_efficiency(samples_used):
    """
    Analyze the efficiency of subsampling in relation to probabilities
    
    Parameters:
    - samples_used: Percentage of samples used by subsampling MH
    
    Returns:
    - Plotly figure
    """
    if samples_used is None or len(samples_used) == 0:
        print("No subsampling data available.")
        return
    
    # Create interactive figure    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution of percentage of samples used',
                        'Evolution of subsampling'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=samples_used,
            nbinsx=20,
            marker_color='skyblue',
            marker_line_color='black',
            marker_line_width=1
        ),
        row=1, col=1
    )
    
    # Temporal evolution
    fig.add_trace(
        go.Scatter(
            x=list(range(len(samples_used))),
            y=samples_used,
            mode='lines',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    # Add line for mean
    fig.add_trace(
        go.Scatter(
            x=[0, len(samples_used)],
            y=[np.mean(samples_used), np.mean(samples_used)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Mean: {np.mean(samples_used):.2f}%'
        ),
        row=1, col=2
    )
    
    # Update axis titles
    fig.update_xaxes(title_text='Percentage of samples used', row=1, col=1)
    fig.update_xaxes(title_text='Iteration', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Percentage of samples used', row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title='Analysis of subsampling efficiency',
        height=500,
        width=1200
    )
    
    # Display results
    print(f"Average percentage of samples used: {np.mean(samples_used):.2f}%")
    print(f"Median: {np.median(samples_used):.2f}%")
    print(f"Min: {np.min(samples_used):.2f}%, Max: {np.max(samples_used):.2f}%")
    
    return fig


def run_probability_analysis(data=None):
    """
    Function running the probability analysis
    
    Parameters:
    - data: Data array (optional)
    """
    print("\n" + "="*50)
    print("CONDITIONAL PROBABILITY ANALYSIS")
    print("="*50 + "\n")
    
    # If no data provided, create test data
    if data is None:
        # Use exponential data
        np.random.seed(42)  # For reproducibility
        data = np.random.exponential(scale=2, size=200)
    
    # Parameters for analysis
    t = 15  # Subsample size
    mean = data.mean()
    print(f"Data mean: {mean:.4f}")
    print(f"Subsample size for probability analysis: {t}")
    
    # Define grids for a and epsilon
    a = np.arange(mean - 30, mean + 30, 1.0)  # Values around the mean
    epsilon = np.arange(0.2, 20., 0.5)  # Proximity thresholds
    
    print("Computing probabilities... (this may take a moment)")
    
    # Create 3D interactive visualization
    fig3d, proba_values, proba_cond_values, X, Y = create_interactive_probability_analysis(data)
    fig3d.show()
    
    # Create complementary 2D visualizations
    fig_2d = create_2d_probability_views(proba_values, proba_cond_values, epsilon, a)
    fig_2d.show()
    
    # Create heatmap
    fig_heatmap = create_probability_heatmap(X, Y, proba_cond_values, mean)
    fig_heatmap.show()


if __name__ == "__main__":
    run_probability_analysis()
