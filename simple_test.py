import numpy as np
import scipy.stats as stats
from MH_algorithm import Kernel, MH_bayesian, MH_bayesian_subsampling
import math



# generating data
true_alpha = 12
true_beta = 5
n = 80
data = stats.gamma(true_alpha, scale = true_beta).rvs(n)


#defining prior as exponential on both
def prior_density(theta):
    # size m x 2
    # we choose as priori a gamma(1,1) for both parameters
    return np.prod(np.exp(-theta), axis = 1)


#defining a log-gaussian transition kernel
class GaussianKernel(Kernel):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def sample(self, theta):
        log_theta = np.log(theta)
        proposition = log_theta + stats.norm().rvs(size = np.shape(theta))*self.sigma
        return np.exp(proposition)

    def density(self, theta_proposition, theta_k):
        """
        The input is m x 2
        The output is 2
        """
        log_theta_propo = np.log(theta_proposition)
        log_theta_k = np.log(theta_k)
        density = np.exp(-((log_theta_propo- log_theta_k)**2)/(2*(self.sigma**2)))/(math.sqrt(2*math.pi)*self.sigma*theta_proposition)
        return np.prod(density, axis = 1)

#defining likelihood
def likelihood(x, theta):
    # Takes data of size n x 1 , m x 2
    # The output is of size n x m
    return stats.gamma.pdf(x[:, np.newaxis], theta[:, 0][ np.newaxis, :], scale= theta[:, 1][np.newaxis, :])



sample_length = 1000

n_iter = 100

sigma = 1

kernel = GaussianKernel(1)

theta_0 = stats.gamma(1,1).rvs(size = (sample_length, 2))

MH_result = MH_bayesian(sample_length = sample_length,  n_iter = n_iter, kernel = kernel, prior_density = prior_density, likelihood = likelihood, data = data, theta_0 = theta_0)
