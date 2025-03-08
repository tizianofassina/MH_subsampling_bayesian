import numpy as np
import scipy.stats as stats
from typing import Callable
import math


class Kernel:
    def __init__(self, **params):
        """
        General stateful proposal kernel.
        Parameters:
        - **params: Dictionary of kernel parameters (e.g., sigma for Gaussian).
        """
        self.params = params
    def sample(self, theta_k: np.ndarray) -> np.ndarray:
        """Generate a new sample given theta (to be implemented by subclasses)."""
        raise NotImplementedError
    def density(self, theta_proposition: np.ndarray, theta_k: np.ndarray) -> np.ndarray:
        """Compute the density p(theta_proposition | theta_k) (to be implemented by subclasses)."""
        raise NotImplementedError

def MH_bayesian(sample_length : int, n_iter : int, kernel : Kernel, prior_density : Callable[[np.ndarray], np.ndarray], likelihood : Callable[[np.ndarray], np.ndarray], data : np.ndarray, theta_0 : np.ndarray) -> list:
    """
    The functions return an array of simulation of size sample_length approximating the posterior law.
    Theta_0 has m rows, each one is a starting point for the algorithm.
    Data has n rows, each one is a data point.
    Likelihood takes has input a data array and a array of parameters and outputs a matrix of size n x m.
    Prior_density takes as input a array of parameters and outputs an array of size m.
    Noyau is used for evaluation and for simulation.
    """
    theta_k = [theta_0]

    for i in range(0, n_iter):

        u = stats.uniform(0,1).rvs( size = sample_length)

        theta_proposition = kernel.sample(theta_k[-1])

        psi = np.log(u * (prior_density(theta_proposition)*kernel.density(theta_proposition, theta_k[-1]))/((prior_density(theta_k[-1])*kernel.density(theta_k[-1], theta_proposition))))
        lambd = np.sum(np.log(likelihood(data, theta_proposition))/np.log(likelihood(data, theta_k[-1])), axis = 0)

        bools = lambd>psi

        theta_new = theta_k[-1].copy()
        theta_new[bools] = theta_proposition[bools]

        theta_k.append(theta_new)

    return theta_k


def MH_bayesian_subsampling(sample_length : int, gamma : float, C: Callable[[np.ndarray], np.ndarray], n_iter : int, kernel : Kernel, prior_density : Callable[[np.ndarray], np.ndarray], likelihood : Callable[[np.ndarray], np.ndarray], data : np.ndarray, theta_0 : np.ndarray, delta : np.ndarray) -> list:
    """
    The functions return an array of simulation of size sample_length approximating the posterior law.
    Theta_0 has m rows, each one is a starting point for the algorithm.
    Data has n rows, each one is a data point.
    Likelihood takes has input a data array and a array of parameters and outputs a matrix of size n x m.
    Prior_density takes as input a array of parameters and outputs an array of size m.
    Noyau is used for evaluation and for simulation.
    The subsampling is managed in a way that allows only the necessary computation. The while loop continues only for the values of lambda star that don't meet the condition yet.
    """
    theta_k = [theta_0]

    for i in range(0, n_iter):

        u = stats.uniform(0,1).rvs( size = sample_length)

        theta_proposition = kernel.sample(theta_k[-1])

        psi = np.log(u * (prior_density(theta_proposition)*kernel.density(theta_proposition, theta_k[-1]))/((prior_density(theta_k[-1])*kernel.density(theta_k[-1], theta_proposition))))

        n = np.shape(data)[0]
        t = 0
        t_look = 0
        lambd = 0

        b = 1
        not_done = np.full(np.shape(theta_0)[0], True)

        data_sampled = np.empty((0, data.shape[1]))
        data_to_be_sampled = data

        while not_done.any():

            sampled_lines = np.random.choice(np.arange(1,np.shape(data_to_be_sampled)[0]), size = b, replace = False)
            data_sampled = np.vstack(data_sampled, data_to_be_sampled[sampled_lines])
            data_to_be_sampled = np.delete(data, sampled_lines, axis = 0)

            lambd[not_done] = (1/b)*(t*lambd[not_done] + np.sum(np.log(likelihood(data_to_be_sampled[sampled_lines], theta_proposition))/np.log(likelihood(data_to_be_sampled[sampled_lines], theta_k[-1])), axis = 0)[not_done])
            t = b
            c  = 2 * C(theta_proposition,theta_k[-1])*math.sqrt(((1-(t-1/n))*math.log(2/delta[t_look]))/(2*t))
            t_look += 1
            b = min(n, math.ceil(gamma*t))


            not_done  = (np.abs(lambd - psi)<c) & (b<=n)

        accepted = (lambd>psi)
        theta_new = theta_k[-1].copy()
        theta_new[accepted] = theta_proposition[accepted]

        theta_k.append(theta_new)

    return theta_k