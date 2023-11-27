import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

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
