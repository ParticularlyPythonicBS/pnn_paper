from PNN import PNN
from sklearn import linear_model
from Utilities import *


# Calculate lyapunov exponent(s)
# ------------------------------------------------------------
# r             range or value of r to calculate the Lyapunov exponent using the method of Rosenstein et al.
# L             the length of time series to generate after the initial transient period
# x0_sep        standard deviation of the initial separation of counterpart trajectories
# NN_Model      filename of the neural network to load. False if calculating the exponent for the logistic map
def lyap(r, L, T, x0_sep, NN_Model=False):
    n_traj = L - T + 1

    # Manage behaviour if r is an array
    if not is_iterable(r):
        rs = np.array([r])
        return_array = False
    else:
        rs = r
        return_array = True

    # Load neural network if calculating the Lyapunov exponent for the neural network
    if NN_Model:
        NN = PNN()
        NN.loadNN(NN_Model)

    # Iterate over r and calculate the Lyapunov exponents
    lyaps = np.zeros(len(rs))
    for index_r, r in enumerate(rs):
        print('Iteration ' + str(index_r+1) + ' of ' + str(len(rs)))
        logds = []

        # Generate time series without the initial transient period
        if NN_Model:
            X0 = GenerateTS(2, 0.5, r)
            X = NN.RollingPrediction(X0, 500+L)[498:]
            ppred = NN.get_parameters(X0)
        else:
            X = GenerateTS(500+L, 0.5, r)[500:]

        # Iterate over the number of trajectories for which the Lyapunov exponent will be calculated
        for traj_index in range(n_traj):
            # try to stay within attractor for the counterpart trajectory
            sep = np.random.rand() * x0_sep
            if X[traj_index] > 0.5:
                sep = -sep

            # X2 is the counterpart trajectory to X1
            X1 = X[traj_index:traj_index+T]
            x0 = X[traj_index] + sep
            if NN_Model:
                X2 = np.concatenate((np.array([x0]), NN.RollingPrediction_with_p(x0, ppred, T)))
            else:
                X2 = GenerateTS(T, x0, r)

            # calculate the distance between trajectories
            d = np.abs(X1 - X2)

            # cut off trajectories that have merged, and discard them if the trajectory is too short
            try:
                same_value_index = np.where(d==0)[0][0]
            except (TypeError, IndexError) as _:
                same_value_index = None
            if same_value_index is not None:
                d = d[:same_value_index]
                if len(d) < 3:
                    continue

            # calculate the log distance between trajectories
            logds.append(np.log(d))

        # Use a smaller number of time steps to calculate the Lyapunov exponent if too many trajectories
        #   have been discarded
        logds_ = []
        T_trunc = T                                         # Number of time steps to use
        for i in range(T):
            logd = 0
            n = n_traj
            for j in range(len(logds)):
                try:
                    logd += logds[j][i]
                except IndexError:
                    n -= 1
            if n < L / 3:
                T_trunc = i
                break
            logds_.append(logd/n)
        logds = np.array(logds_)

        # Estimate the Lyapunov exponent
        t = np.arange(T_trunc)
        reg = linear_model.RANSACRegressor()
        reg.fit(t.reshape(-1, 1), logds.reshape(-1, 1))
        lyaps[index_r] = np.float(reg.estimator_.coef_)

    if NN_Model:
        NN.close()
    if not return_array:
        return np.float(lyaps)
    else:
        return lyaps
