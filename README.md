# Towards scaling up Markov chain Monte Carlo: an adaptive subsampling approach




The article “Towards Scaling Up Markov Chain Monte Carlo: An Adaptive Subsampling Approach” presents a computationally efficient alternative to traditional Metropolis-Hastings sampling by leveraging subsampling. Since standard MCMC methods are computationally expensive, optimizing them is a key research area. This algorithm, designed within a Bayesian framework, approximates posterior distributions by evaluating proposed samples using only a subset of data.
We offer here an implementation of the classical MH algorithm, of the subsampling algorithm and some numerical experiments 

## Repository Structure

`MH_algorithm.py`
This script contains the implementation of the two algorithms and of the necessary functions to use a MH sampler in a bayesian context.

`covtype_test.py`
It implements experiments using a dataset used also in the article

`theoretical_probability_analysis.py`
We implement here some simple plot to show the theoretical problem explained in the teoretical part of our text.



## 🔗 Links  
- **Repository GitHub**: [https://github.com/tizianofassina/MH_subsampling_bayesian.git](https://github.com/tizianofassina/MH_subsampling_bayesian.git)  
- **Articolo originale**: [*Towards scaling up Markov chain Monte Carlo: an adaptive subsampling approach*](https://proceedings.mlr.press/v32/bardenet14.html)