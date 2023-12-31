import numpy as np

def moving_average(X, window):
    '''
    Applies a moving average filter to the input signal.

    Args:
        X: The input signal (PyTorch tensor).
        window: The window of the moving average filter.

    Returns:
        y: The output signal after applying the moving average filter.
    '''

    N = X.shape[0]

    if window is None:
      window = np.ones((np.arange(N), 1))

    len_window = window.shape[0]

    y = np.empty_like(X)

    for i in range(X.shape[0]):
        is_odd = int(np.mod(len_window, 2) == 1)

        m = np.arange((i - (len_window - is_odd) / 2), (i + (len_window - is_odd) / 2 - (is_odd == 0) + 1), dtype = np.int32)
        
        k = m[(m >= 0) & (m < N)]
    
        window_ = window[(m >= 0) & (m < N)]
        
        window_ /= window_.sum(0)

        y[i] = np.matmul(window_, X[k])

    return y
