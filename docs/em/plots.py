import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()
# ax = plt.axes(projection="3d")


# theta = np.linspace(0, 20, 30)
# tau = np.linspace(0, 20, 30)
# Theta, Tau = np.meshgrid(theta, tau)
# Z = -((Tau - Theta) ** 2) * (Theta ** 2 + 1)

# ax.plot_surface(Theta, Tau, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
# ax.set_xlabel(r"$\theta$")
# ax.set_ylabel(r"$\tau$")
# plt.show()

f = (
    lambda theta, tau: -((tau - np.sum(theta, axis=-1)) ** 2)
    + 3 / 2 * np.sum(theta, axis=-1) ** 2
)

fig, ax = plt.subplots()
tau = np.linspace(0, 20, 100)
ax.plot(tau, f([1, 2], tau))
ax.plot(tau, f([2, 3], tau))
ax.plot(tau, f([3, 4], tau))

# fig, ax = plt.subplots()
# theta = np.linspace(-100, 100, 100)
# ax.plot(theta, f(theta, 0))
# ax.plot(theta, f(theta, 10))


fig = plt.figure()
ax = plt.axes(projection="3d")
theta = np.linspace(0, 20, 60)
tau = np.linspace(0, 20, 60)
Theta, Tau = np.meshgrid(theta, tau)
Z = f(Theta.reshape(60,60,1), Tau)
ax.plot_surface(Theta, Tau, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\tau$")
plt.show()