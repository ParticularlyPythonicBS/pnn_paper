from Lyapunov import lyap
import numpy as np
from matplotlib import pyplot as plt

# Plot the Lyapunov exponents of the logistic map and the neural network (Fig. 3d of the paper)

# Define range of r over which to plot
rs = np.linspace(2.5, 4, 3000, endpoint=False)
#rs = np.linspace(3.565, 3.572, 200)            # Used to estimate the onset of chaos
#x0_sep = 0.0001                                # Used to estimate the onset of chaos

x0_sep = 0.001
r_test_index = np.where(rs > 3.4)[0][0]         # index of r where the prediction regime starts
seed = 1
L = 200
T = 5

lyaps_NN = lyap(rs, L, T, x0_sep, NN_Model=seed)            # Load the neural network trained with seed 1
lyaps_Actual = lyap(rs, L, T, x0_sep, NN_Model=False)
rs_train = rs[:r_test_index]
rs_test = rs[r_test_index:]
lyaps_NN_train = lyaps_NN[:r_test_index]
lyaps_NN_test = lyaps_NN[r_test_index:]
plt.figure(figsize=(12,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.plot(rs_train, lyaps_NN_train, 'b', label='Neural network\ntraining regime')
plt.plot(rs_test, lyaps_NN_test, 'r', label='Neural network\ntesting regime')
plt.plot(rs, lyaps_Actual, 'k--', label='Logistic map\nnumerical estimates')
plt.xlabel('$r$', fontsize=28)
plt.ylabel('Lyapunov exponent, $\lambda$', fontsize=28)
plt.axhline(y=0, color='k')
plt.legend()
plt.tight_layout()
plt.savefig('LyapunovExponent.png', dpi=200)
plt.show()
