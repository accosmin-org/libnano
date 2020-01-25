import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, +5, 1000)

# Classification loss functions
y_error = (1 - np.sign(x)) / 2
y_hinge = np.maximum(1 - x, 0)
y_savage = 1 / np.square(1 + np.exp(x))
y_tangent = np.square(2 * np.arctan(x) - 1)
y_logistic = np.log(1 + np.exp(-x))
y_exponential = np.exp(-x)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([-5.0,+5.0])
ax.set_ylim([-0.3,+8.0])
ax.set_title("classification loss functions")
ax.set_xlabel("edge = target * output")
ax.set_ylabel("loss(edge)")
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.plot(x, y_error, 'k', label="0-1 loss", linewidth=2, linestyle="--")
plt.plot(x, y_hinge, 'r', label="hinge", linewidth=2)
plt.plot(x, y_savage, 'm', label="savage", linewidth=2)
plt.plot(x, y_tangent, 'c', label="tangent", linewidth=2)
plt.plot(x, y_logistic, 'g', label="logistic", linewidth=2)
plt.plot(x, y_exponential, 'b', label="exponential", linewidth=2)

plt.legend(loc='upper right')
plt.savefig("plot_losses_classification.png")
plt.clf()


# Regression loss functions
y_absolute = np.abs(x)
y_squared = np.square(x) / 2
y_cauchy = np.log(1 + np.square(x)) / 2

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([-5.0,+5.0])
ax.set_ylim([-0.3,+8.0])
ax.set_title("regression loss functions")
ax.set_xlabel("diff = target - output")
ax.set_ylabel("loss (diff)")
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.plot(x, y_absolute, 'r', label="absolute", linewidth=2)
plt.plot(x, y_squared, 'g', label="squared", linewidth=2)
plt.plot(x, y_cauchy, 'b', label="cauchy", linewidth=2)

plt.legend(loc='upper right')
plt.savefig("plot_losses_regression.png")
