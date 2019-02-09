import tensorflow as tf
from Utilities import *
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pickle

# Define neural network architecture and training routines

class PNN:

    def __init__(self):
        self.W = None
        self.seed = None
        self.NN = None
        self.look_back = None
        self.look_ahead = None
        self.p_layers = None
        self.m_layers = None
        self.p_units = None
        self.m_units = None
        self.train_step = None
        self.batchX_placeholder = None
        self.batchY_placeholder = None
        self.Xpred_placeholder = None
        self.ppredroll_placeholder = None
        self.Xpredroll_placeholder = None
        self.Ypred = None
        self.batch_size = None
        self.optimizer = None
        self.ppred = None
        self.e = None
        self.T = None
        self.N_windows = None
        self.ys = None
        self.p = None
        self.Wps = None
        self.bps = None
        self.Wms = None
        self.bms = None
        self.Wpout = None
        self.bpout = None
        self.Wout = None
        self.bout = None
        self.sess = None
        self.ax11 = None
        self.ax21 = None
        self.ax12 = None
        self.ax22 = None
        self.ax3 = None
        self.dropout = None

    # set neural network architecture
    # ------------------------------------------------------------
    # look_back         number of lags
    # look_ahead        number of predictive time steps to train
    # p_units           number of neurons in each layer of the parameterization layers
    # m_units           number of neurons in each layer of the modeling layers
    # p_layers          number of parameterization layers
    # m_layers          number of modeling layers
    # batch_size        number of timeseries in each minibatch
    # T                 length of training time series generated for each x0 and r before slicing into shorter
    #                       overlapping time series
    # seed              seed used for the random number generator
    # W                 initialize weights, will be overwritten if training is initiated
    def SetArchitecture(self, look_back, look_ahead, p_units, m_units,
                        p_layers, m_layers, batch_size, T, seed=None, W=None, precision=tf.float32):
        precision = precision
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.set_architecture_attributes(look_back, look_ahead, p_units, m_units, p_layers, m_layers,
                                         batch_size, T, W, seed)
        self.N_windows = T - look_back
        initializer = tf.contrib.layers.xavier_initializer(dtype=precision)
        zero_initializer = tf.zeros_initializer(dtype=precision)
        self.batchX_placeholder = tf.placeholder(precision)
        self.batchY_placeholder = tf.placeholder(precision)
        self.ppredroll_placeholder = tf.placeholder(precision)
        self.Xpred_placeholder = tf.placeholder(precision)
        self.Xpredroll_placeholder = tf.placeholder(precision)
        Mp = np.full(p_layers, None)
        Mo = np.full(m_layers, None)
        self.Wps, self.bps = np.copy(Mp), np.copy(Mp)
        self.Wms, self.bms = np.copy(Mo), np.copy(Mo)
        self.Wout = tf.Variable(initializer((m_units, 1)), dtype=precision)
        self.bout = tf.Variable(zero_initializer(1), dtype=precision)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1E-5)
        for i in range(p_layers):
            input_length = p_units
            if i == 0:
                input_length = look_back
            self.Wps[i] = tf.Variable(initializer((input_length, p_units)), dtype=precision)
            self.bps[i] = tf.Variable(zero_initializer((1, p_units)))
        for i in range(m_layers):
            input_length = m_units
            if i == 0:
                input_length = p_units + look_back - 1
            self.Wms[i] = tf.Variable(initializer((input_length, m_units)), dtype=precision)
            self.bms[i] = tf.Variable(zero_initializer((1, m_units)))

        # parameterize inputs
        p = self.parameterize(self.batchX_placeholder)
        # modeling inputs together with parameters
        self.ys = []
        for t in range(look_ahead+look_back-1):
            if t == 0:
                x = self.batchX_placeholder[:, :-1]
            elif t == 1:
                x = self.batchX_placeholder[:, 1:]
            else:
                x = self.ys[-1]
            self.ys.append(self.model(p, x))

        # backprop with adam
        self.ys = tf.concat(self.ys, axis=1)
        self.e = tf.reduce_mean(tf.squared_difference(self.batchY_placeholder, self.ys))
        self.train_step = self.optimizer.minimize(self.e)

        # calculate parameters
        self.ppred = self.parameterize(tf.reshape(self.Xpred_placeholder, (1, look_back)))

        # prediction with input parameters and x0
        self.Ypred = []
        self.Ypred = self.model(self.ppredroll_placeholder, tf.reshape(self.Xpredroll_placeholder, (1, look_back-1)))
        self.Ypred = self.Ypred[0, 0]

        if W is not None:
            self.set_weights(W)

    # paramterization layers
    def parameterize(self, x):
        h = None
        for layer_index in range(self.p_layers):
            if layer_index == 0:
                h = tf.tanh(tf.matmul(x, self.Wps[layer_index]) + self.bps[layer_index])
            else:
                h = tf.tanh(tf.matmul(h, self.Wps[layer_index]) + self.bps[layer_index])
        p = h
        return p

    # modeling layers
    def model(self, p, x):
        input_o = tf.concat((p, x), 1)
        h = None
        for layer_index in range(self.m_layers):
            if layer_index == 0:
                h = tf.tanh(tf.matmul(input_o, self.Wms[layer_index]) + self.bms[layer_index])
            else:
                h = tf.tanh(tf.matmul(h, self.Wms[layer_index]) + self.bms[layer_index])
            if self.dropout:
                h = tf.nn.dropout(h, self.dropout)
        y = tf.matmul(h, self.Wout) + self.bout
        return y

    # fit the neural network
    # ------------------------------------------------------------
    # xtrain                    input training data
    # ytrain                    target training data
    # data_val                  validation data (for observing and plotting the validation error)
    # x0_xval                   initial x to use when plotting forecasted time series for validation
    # r_xval                    r value to use when plotting forecasted time series for validation
    # epochs                    number of epochs
    # x0_xval2                  another initial x to use when plotting forecasted time series for validation
    # r_xval2                   another r value to use when plotting forecasted time series for validation
    def Fit(self, xtrain, ytrain, data_val, x0_xval, r_xval, epochs=100, x0_xval2=None, r_xval2=None, plot=True):
        n_train_samples = np.shape(xtrain)[0]
        num_batches = int(np.ceil(n_train_samples/self.batch_size))
        xtest = data_val[0]
        ytest = data_val[1]
        self.start_session()
        self.sess.run(tf.global_variables_initializer())
        plt.figure(figsize=(15,8.5))
        plt.ion()
        losses = []
        val_losses = []
        print('Train on ' + str(n_train_samples) + ' samples, validiate on ' + str(np.shape(xtest)[0]) + ' samples')
        ts2 = TimeScript()
        for epoch_index in range(1, epochs+1):
            ts1 = TimeScript()
            print('Epoch ' + str(epoch_index) + '/' + str(epochs))
            shuffled_indices = np.arange(n_train_samples)
            np.random.shuffle(shuffled_indices)
            x_train_epoch = xtrain[shuffled_indices]
            y_train_epoch = ytrain[shuffled_indices]
            batch_losses = []
            for batch_index in range(1, num_batches+1):
                if batch_index == num_batches:
                    batch_size = n_train_samples - (batch_index-1)*self.batch_size
                else:
                    batch_size = self.batch_size
                start_index = (batch_index-1) * self.batch_size
                end_index = start_index + batch_size
                x_batch = x_train_epoch[start_index:end_index]
                y_batch = y_train_epoch[start_index:end_index]
                fd = {self.batchX_placeholder: x_batch, self.batchY_placeholder: y_batch}
                loss, _train_step, = self.sess.run(
                    [self.e, self.train_step],
                    feed_dict=fd
                )
                batch_losses.append(loss)
            losses.append(np.mean(batch_losses))
            fd = {self.batchX_placeholder: xtest, self.batchY_placeholder: ytest}
            val_loss = self.sess.run(self.e, feed_dict=fd)
            val_losses.append(val_loss)
            secs_elapsed = ts1.clock_seconds()
            print(' - ' + '{:.2f}'.format(secs_elapsed) + 's - loss: ' + num_format(losses[-1]) +
                  ' - val_loss: ' + num_format(val_loss))
            if plot and ts2.clock_seconds() > 5:
                self.plot(losses, val_losses, x0_xval, r_xval, x0_xval2, r_xval2)
                ts2.reset()

    # neural network forecasting based on a starting xseed (array of length 2)
    def RollingPrediction(self, xseed, T):
        self.start_session()

        ppred = self.sess.run(self.ppred, feed_dict={self.Xpred_placeholder: xseed})
        N_windows = T - self.look_back
        X = []
        x = xseed[-1]
        for t in range(N_windows):
            x = self.sess.run(self.Ypred, feed_dict={self.ppredroll_placeholder: ppred, self.Xpredroll_placeholder: x})
            X.append(x)
        return np.array(X)

    # get parameters of xseed using the parameterization layers
    def get_parameters(self, xseed):
        self.start_session()
        ppred = self.sess.run(self.ppred, feed_dict={self.Xpred_placeholder: xseed})
        return ppred

    # neural network forecasting based on a starting point (x0) and parameters (ppred)
    def RollingPrediction_with_p(self, x0, ppred, T):
        N_windows = T - self.look_back
        X = []
        x = x0
        for t in range(N_windows + 1):
            x = self.sess.run(self.Ypred, feed_dict={self.ppredroll_placeholder: ppred, self.Xpredroll_placeholder: x})
            X.append(x)
        return np.array(X)

    # get the weights of the neural network
    def get_weights(self):
        self.start_session()
        return self.sess.run([list(self.Wps), list(self.bps), list(self.Wms), list(self.bms), self.Wout, self.bout])

    # set the weights of the neural network
    def set_weights(self, W):
        self.start_session()
        self.assign_layer_weights(self.Wps, W[0])
        self.assign_layer_weights(self.bps, W[1])
        self.assign_layer_weights(self.Wms, W[2])
        self.assign_layer_weights(self.bms, W[3])
        self.sess.run(tf.assign(self.Wout, W[4]))
        self.sess.run(tf.assign(self.bout, W[5]))

    # assign weights for a block layer by layer
    def assign_layer_weights(self, F, G):
        l = len(F)
        for i in range(l):
            self.sess.run(tf.assign(F[i], G[i]))

    # start tensorflow session
    def start_session(self):
        if self.sess is None or self.sess._closed:
            self.sess = tf.Session()

    # close tensorflow session
    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    # save neural network weights and training parameters
    def saveNN(self, file_name=None):
        if file_name is None:
            file_name = 'temp'
        W = self.get_weights()
        saved_data = {'W': W, 'p_layers': self.p_layers, 'm_layers': self.m_layers, 'p_units': self.p_units,
                      'm_units': self.m_units, 'look_back': self.look_back, 'look_ahead': self.look_ahead,
                      'seed': self.seed, 'batch_size': self.batch_size, 'T': self.T}
        path = 'NN Models\\' + str(file_name) + '.model'
        with open(path, 'wb') as f:
            pickle.dump(saved_data, f)

    # load neural network weights and training parameters
    def loadNN(self, file_name=None, T=None, precision64=False):
        if file_name is None:
            file_name = 'temp'
        path = 'NN Models\\' + str(file_name) + '.model'
        with open(path, 'rb') as f:
            saved_data = pickle.load(f)
        if T is not None:
            saved_data['T'] = T
        if precision64 is True:
            saved_data['precision'] = tf.float64
        self.SetArchitecture(**saved_data)

    # set class attributes
    def set_architecture_attributes(self, look_back, look_ahead,
                                    p_units, o_units, p_layers, o_layers, batch_size, T, W, seed):
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.p_units = p_units
        self.m_units = o_units
        self.p_layers = p_layers
        self.m_layers = o_layers
        self.batch_size = batch_size
        self.T = T
        self.W = W
        self.seed = seed

    # plot the training error, validation error, and forecasted time series
    #   top left and top right: training error (blue) and validation error (black)
    #   middle and bottom: validation forecasted time series and actual time series
    def plot(self, losses, val_losses, x0_xval, r_xval, x0_xval2, r_xval2):
        if x0_xval2 is None:
            gs = gridspec.GridSpec(2, 2)
        else:
            gs = gridspec.GridSpec(3, 2)
        ax11 = plt.subplot(gs[0, 0])
        ax12 = ax11.twinx()
        trailing_epochs = 5000
        X, Y, x = SlidingWindows(self.T, self.look_back, 1, x0_xval, r_xval)
        xroll_pred = self.RollingPrediction(X[0, :], self.T)

        t1 = 1000 if len(losses) > 1200 else 1
        t1 = max(t1, len(losses) - 50000 + 1)
        t = np.arange(t1, len(losses) + 1)
        ax11.cla()
        ax11.plot(t, losses[t1-1:], 'b-')
        ax11.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax11.tick_params('y', colors='b')
        ax12.cla()
        ax12.plot(t, val_losses[t1-1:], 'k-')
        ax12.tick_params('y', colors='k')
        ax12.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax21 = plt.subplot(gs[0, 1])
        ax22 = ax21.twinx()
        ax21.cla()
        ax22.cla()
        t1 = max(1, len(losses)-trailing_epochs+1)
        t = np.arange(t1, len(losses) + 1)
        ax21.set_xlim(t[0], t[-1])
        ax21.plot(t, losses[t1-1:], 'b-')
        ax21.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax21.tick_params('y', colors='b')
        ax22.plot(t, val_losses[t1-1:], 'k-')
        ax22.tick_params('y', colors='k')
        ax22.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        if x0_xval2 is not None:
            ax3 = plt.subplot(gs[1, 0])
            ax3.cla()
            t = np.arange(self.look_back + 1, self.T + 1)
            ax3.plot(t, xroll_pred, 'k-', label='Neural Network')
            ax3.plot(t, x[self.look_back:], 'b--', label='Actual')
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_xlim(t[0], t[-1])

            X2, Y2, x2 = SlidingWindows(self.T, self.look_back, 1, x0_xval2, r_xval)
            xroll_pred2 = self.RollingPrediction(X2[0, :], self.T)
            ax4 = plt.subplot(gs[1, 1])
            ax4.cla()
            t = np.arange(self.look_back + 1, self.T + 1)
            ax4.plot(t, xroll_pred2, 'k-', label='Neural Network')
            ax4.plot(t, x2[self.look_back:], 'b--', label='Actual')
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_xlim(t[0], t[-1])

            X2, Y2, x2 = SlidingWindows(self.T, self.look_back, 1, x0_xval, r_xval2)
            xroll_pred2 = self.RollingPrediction(X2[0, :], self.T)
            ax5 = plt.subplot(gs[2, 0])
            ax5.cla()
            t = np.arange(self.look_back + 1, self.T + 1)
            ax5.plot(t, xroll_pred2, 'k-', label='Neural Network')
            ax5.plot(t, x2[self.look_back:], 'b--', label='Actual')
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.set_xlim(t[0], t[-1])

            X2, Y2, x2 = SlidingWindows(self.T, self.look_back, 1, x0_xval2, r_xval2)
            xroll_pred2 = self.RollingPrediction(X2[0, :], self.T)
            ax6 = plt.subplot(gs[2, 1])
            ax6.cla()
            t = np.arange(self.look_back + 1, self.T + 1)
            ax6.plot(t, xroll_pred2, 'k-', label='Neural Network')
            ax6.plot(t, x2[self.look_back:], 'b--', label='Actual')
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.set_xlim(t[0], t[-1])
        else:
            ax3 = plt.subplot(gs[1, :])
            ax3.cla()
            t = np.arange(self.look_back + 1, self.T + 1)
            ax3.plot(t, xroll_pred, 'k-', label='Neural Network')
            ax3.plot(t, x[self.look_back:], 'b--', label='Actual')
            ax3.set_xticks([])
            ax3.set_xlim(t[0], t[-1])

        plt.tight_layout()
        plt.pause(0.01)
