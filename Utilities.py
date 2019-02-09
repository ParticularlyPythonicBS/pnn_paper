import numpy as np
import time
from LogisticMap import GenerateTS

class TimeScript():
    def __init__(self):
        self.t1 = time.time()
        self.t2 = None

    def clock(self):
        print()
        self.t2 = time.time()
        dt = self.t2 - self.t1
        minute, second = divmod(dt, 60)
        hour, minute = divmod(minute, 60)
        day, hour = divmod(hour, 24)
        print('Time taken: ', end='')
        b = PrintTimeUnit(day, 'day')
        b = PrintTimeUnit(hour, 'hour', b)
        b = PrintTimeUnit(minute, 'minute', b)
        if b:
            print('and ', end='')
        PrintTimeUnit(second, 'second', b)

    def reset(self):
        self.t1 = time.time()

    def clock_seconds(self):
        self.t2 = time.time()
        return self.t2 - self.t1


# Generate sliding windows from the logistic map
# ---------------------------------------------------
# T             length of time series to generate
# look_back     number of lags
# look_ahead    number of predictive time steps (length of sliding windows = look_back + look_ahead)
# x0            starting initial point
# r             r parameter of the logistic map
def SlidingWindows(T, look_back, look_ahead, x0=None, r=None):
    N_windows = T - look_back - look_ahead + 1
    X = np.zeros((N_windows, look_back))
    Y = np.zeros((N_windows, look_ahead))
    x = GenerateTS(T, x0, r)
    for i in range(N_windows):
        X[i, :] = x[i:i + look_back]
        Y[i, :] = x[i + look_back:i + look_back + look_ahead]
    return X, Y, x


def is_iterable(x):
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def shuffle_mini_batch_indices(n_train_samples, n_rs, N_windows):
    indices = np.arange(n_train_samples)
    assert(n_train_samples%n_rs == 0)
    n_xs = int(n_train_samples / n_rs)
    n_x0s = int(n_xs/N_windows)
    batch_size = N_windows*n_rs
    indices_ = np.zeros(n_train_samples, dtype='int')
    for i in range(n_x0s):
        for j in range(N_windows):
            indices_[i*batch_size+j*n_rs : i*batch_size+(j+1)*n_rs] = indices[i*N_windows+j::n_xs]
    r_indices = np.arange(n_x0s)
    indices = np.copy(indices_)
    np.random.shuffle(r_indices)
    for i in range(n_x0s):
        indices[i*batch_size : (i+1)*batch_size] = indices_[r_indices[i]*batch_size : (r_indices[i]+1)*batch_size]
        np.random.shuffle(indices[i*batch_size : (i+1)*batch_size])
    return indices


def num_format(x):
    return '{x:.{k}{c}}'.format(x=x, c='f' if x > 1E-3 else 'e', k='3' if x < 1E-3 else '5')


def PrintTimeUnit(q, s, print_zero=False):
    if print_zero or q != 0:
        if s == 'second':
            q = round(q, 1)
        print(str(round(q, 1)) + ' ' + s, end='')
        if q!= 1:
            print('s', end='')
        if s == 'second':
            print('.')
        else:
            print(', ', end='')
        return(True)
    return(False)
