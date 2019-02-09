import os
from PNN import PNN
from Utilities import *

# Train the neural network that is used in the paper

# Training without a GPU should generally be faster for this network
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ts1 = TimeScript()

# Define neural network training parameters and training range
rs = np.arange(0, 3.4+0.02, 0.02)
x0s = np.arange(0, 1 + 0.05, 0.05)

seed = 1
T = 40
epochs = 700000
look_back = 2
batch_size = 1000
p_units = 10
m_units = 10
p_layers = 5
m_layers = 5
x0_xval = 0.2
r_xval = 3.7
x0_xval2 = 0.6
r_xval2 = 3.56
look_ahead = 3

# Create training data and validation set
N_windows = T - look_back - look_ahead + 1
X = np.zeros((len(x0s)*N_windows*len(rs), look_back))
Xval, Yval, xval = SlidingWindows(T, look_back, look_ahead, x0_xval, r_xval)
Yval = np.concatenate((Xval[:, 1:], Yval), axis=1)
Y = np.zeros((len(x0s) * N_windows * len(rs), look_ahead+look_back-1))
count = 0
for r in rs:
    for i in range(len(x0s)):
        x0 = x0s[i]
        X_, Y_, _ = SlidingWindows(T, look_back, look_ahead, x0, r)
        N = np.shape(X_)[0]
        X[count:count+N,] = X_
        Y[count:count+N,] = np.concatenate((X_[:, 1:], Y_), 1)
        count += N

# Train the neural network and save the trained network to a file
NN = PNN()
NN.SetArchitecture(look_back, look_ahead, p_units, m_units, p_layers, m_layers, batch_size, T, seed)
NN.Fit(X, Y, (Xval, Yval), x0_xval, r_xval, epochs, x0_xval2, r_xval2, plot=True)
NN.saveNN(seed)
NN.close()

ts1.clock()
