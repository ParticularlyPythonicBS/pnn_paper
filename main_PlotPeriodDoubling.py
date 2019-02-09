from PNN import PNN
import numpy as np
from LogisticMap import GenerateTS
from matplotlib import pyplot as plt

# Plot the period doubling cascade of the neural network from r=3.43 to r=3.569 (Fig. 3c of the paper)

N = 1500
cutoff = 500
x0 = 0.5
rs = np.linspace(3.43, 3.569, 1000)
seed = 1

NN = PNN()
NN.loadNN(seed)                                                 # Load the neural network trained with seed 1
XsNN = np.zeros((len(rs), N-cutoff))
RsNN = np.ones((len(rs), N-cutoff))
for i, r in enumerate(rs):
    print('Iteration ' + str(i) + ' of ' + str(len(rs)))
    X0 = GenerateTS(2, x0, r)
    RsNN[i, :] *= r
    XsNN[i, :] = NN.RollingPrediction(X0, N)[cutoff-2:]
XsNN = XsNN.flatten()
RsNN = RsNN.flatten()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)

f = plt.figure(figsize=(4.5,8.5))
f.text(0, 0.5, '$X^*$', fontsize=28, va='center', rotation='vertical')

ax1 = plt.subplot(211)
plt.plot(RsNN[XsNN>0.7], XsNN[XsNN>0.7], 'r.', markersize=0.05)
plt.xticks([])
plt.axis([rs[0], rs[-1], 0.8, 0.895])

ax2 = plt.subplot(212)
plt.plot(RsNN[XsNN<0.57], XsNN[XsNN<0.57], 'r.', markersize=0.05)
plt.axis([rs[0], rs[-1], 0.34, 0.57])

plt.xlabel('$r$', fontsize=28)
plt.tight_layout()
plt.gcf().subplots_adjust(left=0.25)
plt.savefig('NNPeriodDoubling.png', dpi=200)
plt.show()
