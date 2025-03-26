import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


random_numbers = np.random.exponential(scale=2, size=200)

mean = random_numbers.mean()


def proba(numbers, epsilon, t, num_samples) :
    sampled_lines = np.array([np.random.choice(np.arange(numbers.shape[0]), size= t, replace=False) for _ in range(num_samples)])
    means = np.mean(numbers[sampled_lines], axis = 1)
    A = np.mean(
        (np.abs(means[:, np.newaxis] - numbers.mean()) < epsilon[np.newaxis,:]),
        axis=0)
    return A


def proba_conditional(numbers, epsilon, t, num_samples, a):
    sampled_lines = np.array(
        [np.random.choice(np.arange(numbers.shape[0]), size=t, replace=False) for _ in range(num_samples)])

    means = np.mean(numbers[sampled_lines], axis=1)

    A = np.mean(
        (np.abs(means[:, np.newaxis, np.newaxis] - a[np.newaxis, :, np.newaxis]) > epsilon[np.newaxis, np.newaxis,
                                                                                   :]) & (
                    np.abs(means[:, np.newaxis, np.newaxis] - numbers.mean()) < epsilon[np.newaxis, np.newaxis,
                                                                                   :]),
        axis=0)

    B = np.mean(np.abs(means[:, np.newaxis, np.newaxis] - a[np.newaxis,:,np.newaxis]) > epsilon[np.newaxis,np.newaxis, :], axis = 0)

    return A / B

data = np.random.uniform(low=0.0, high=1.0, size=300)*200

t = 15

mean = data.mean()

a  = np.arange(mean - 30, mean + 30, 0.1)

epsilon = np.arange(0.2, 20., 0.1)


proba = proba(numbers = data, epsilon = epsilon, t = t, num_samples=2000)
proba_cond = proba_conditional(numbers = data, epsilon = epsilon, t = t, num_samples=2000, a =a )

proba = np.tile(proba, (a.shape[0], 1))
X, Y = np.meshgrid(epsilon, a)

fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, proba, cmap='viridis', alpha=0.9)
ax1.set_title("Probability Surface")
ax1.set_xlabel(r"$\epsilon$")
ax1.set_ylabel('a')
ax1.set_zlabel("Probability")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, proba_cond, cmap='inferno', alpha=0.5)
ax2.set_title("Conditional Probability Surface")
ax2.set_xlabel(r"$\epsilon$")
ax2.set_ylabel('a')
ax2.set_zlabel("Conditional Probability")

plt.tight_layout()
plt.show()