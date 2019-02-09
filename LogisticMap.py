import numpy as np
from matplotlib import pyplot as plt

# Generate time series from the logistic map
# ------------------------------------------------------------
# T                 length of time series
# x0                initial x value of the trajectory
# r                 r value of the logistic map
def GenerateTS(T, x0, r):
    X = np.zeros(T, dtype=np.float64)
    X[0] = x0
    for i in range(1, T):
        X[i] = r * X[i - 1] * (1 - X[i - 1])
    return X


# Plot the bifurcation diagram of the logistic map
# ------------------------------------------------------------
# rs                range of r values to use when plotting
# x0                initial x value of the trajectory
# N                 total length of time series to use (must be bigger than cutoff i.e. >500)
def BifurcationDiagram(rs, x0, N):
    cutoff = 500
    Xs = np.zeros((len(rs), N-cutoff))
    Rs = np.ones((len(rs), N-cutoff))
    for i, r in enumerate(rs):
        Rs[i, :] *= r
        Xs[i, :] = GenerateTS(N, x0, r)[cutoff:]
    Xs = Xs.flatten()
    Rs = Rs.flatten()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=20)
    plt.figure(figsize=(8,6))
    plt.plot(Rs, Xs, 'k.', markersize=0.1)
    plt.axis([rs[0], rs[-1], 0, 1])
    plt.xlabel('$r$', fontsize=28)
    plt.ylabel('$X^*$', fontsize=28)
    plt.tight_layout()
    plt.savefig('LogisticBifurcationDiagram.png', dpi=200)
    plt.show()
