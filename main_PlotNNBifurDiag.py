from PNN import PNN
from LogisticMap import GenerateTS
import numpy as np
from matplotlib import pyplot as plt

# Plot the bifurcation diagram of the neural network (Fig. 3b of the paper)

N = 1500
cutoff = 500
x0 = 0.5
rs = np.linspace(2.5, 4, 1000)
r_test_index = np.where(rs > 3.4)[0][0]
seed = 1

NN = PNN()
NN.loadNN(seed)                                                 # Load the neural network trained with seed 1
Xs = np.zeros((len(rs), N-cutoff))
Rs = np.ones((len(rs), N-cutoff))
for i, r in enumerate(rs):
    print('Iteration ' + str(i) + ' of ' + str(len(rs)))
    X0 = GenerateTS(2, x0, r)
    Rs[i, :] *= r
    Xs[i, :] = NN.RollingPrediction(X0, N)[cutoff-2:]
Xs_train = Xs[:r_test_index].flatten()
Rs_train = Rs[:r_test_index].flatten()
Xs_test = Xs[r_test_index:].flatten()
Rs_test = Rs[r_test_index:].flatten()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.figure(figsize=(8,6))
plt.plot(Rs_train, Xs_train, 'b.', markersize=0.1)
plt.plot(Rs_test, Xs_test, 'r.', markersize=0.1)
plt.axis([rs[0], rs[-1], 0, 1])

plt.plot([3.4, 3.4], [0, 0.16], 'b', linewidth=2.0)
plt.arrow(3.35, 0.14, 3.4-3.35, 0, color='blue', head_width=0.03, overhang=0.4, length_includes_head=True)
plt.plot([rs[0], 3.35], [0.14, 0.14], 'b--', linewidth=2.0)
plt.text(3.38, 0.02, 'Training Regime\n$r_{\mathrm{train}}\in [0, 3.4]$', fontsize=18, color='blue', horizontalalignment='right')
plt.text(3.855, 0.02, 'Prediction Regime\n$r_{\mathrm{test}}\in (3.4, 4]$', fontsize=18, color='red',
                 horizontalalignment='right')
plt.xlabel('$r$', fontsize=28)
plt.ylabel('$X^*$', fontsize=28)
plt.tight_layout()
plt.savefig('NNBifurcationDiagram.png', dpi=200)
plt.show()
