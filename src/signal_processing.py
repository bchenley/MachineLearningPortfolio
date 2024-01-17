import pandas as pd
import numpy as np
import scipy as sc

def periodogram(X, fs=1, window='hann', nfft=None,
                detrend=None, return_onesided=True,
                scaling='density', axis=0):
  '''
  Computes the periodogram of a signal using the SciPy library.

  Args:
      X: The input signal.
      fs: The sampling frequency of the input signal.
      window: The window function to apply to the signal.
      nfft: The number of points to compute the FFT.
      detrend: The detrend function to remove a trend from the signal.
      return_onesided: If True, returns only the one-sided spectrum for real inputs.
      scaling: The scaling mode for the power spectrum.
      axis: The axis along which to compute the periodogram.

  Returns:
      f: The frequencies at which the periodogram is computed.
      psd: The power spectral density (periodogram) of the signal.
  '''
  if nfft is None:  # or nfft < x.shape[dim]
      nfft = X.shape[axis]
      print(f'nfft set to {nfft}')

  f, psd = sc.signal.periodogram(X, fs=fs, window=window, nfft=nfft,
                          detrend=detrend, return_onesided=return_onesided,
                          scaling=scaling, axis=axis)

  return f, psd

def fft(x, fs=1, axes=0, nfft=None, norm='backward'):
    """
    Computes the Fast Fourier Transform (FFT) of the input signal.
    
    Args:
        x (np.ndarray or pd.DataFrame): The input signal.
        fs (float): The sampling frequency of the input signal.
        axes (int or tuple of ints): The dimension(s) along which to compute the FFT.
        nfft (int): The number of FFT points. Defaults to the size of the input signal along the specified axis.
        norm (str): The normalization mode, 'backward' (default) or 'forward'.
    
    Returns:
        freq (np.ndarray): The frequency values corresponding to the FFT.
        x_fft_mag (np.ndarray): The magnitude of the FFT coefficients.
        x_fft_phase (np.ndarray): The phase of the FFT coefficients.
    """
    # Convert pandas DataFrame to numpy array if necessary
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    # Determine the shape of the output FFT
    s = x.shape[axes] if nfft is None else nfft
    
    # Compute the FFT along the specified axis
    
    x_fft = np.fft.fft(x, n=s, axis=axes, norm=norm)
    
    # Calculate the frequencies corresponding to the FFT
    freq = np.fft.fftfreq(n=s, d=1/fs)
    
    # Only take the positive frequencies and corresponding FFT values
    if s % 2 == 0:
        freq = freq[:s//2]
        x_fft = x_fft[:s//2, :]
    else:
        freq = freq[:(s//2 + 1)]
        x_fft = x_fft[:(s//2 + 1), :]

    # Calculate magnitude and phase
    x_fft_mag = np.abs(x_fft)
    x_fft_phase = np.angle(x_fft)
    
    # Adjust the magnitude for normalization ('backward' FFT normalization includes a factor of 1/n)
    if norm == 'backward':
        x_fft_mag *= (2/s)

    return freq, x_fft_mag, x_fft_phase

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
