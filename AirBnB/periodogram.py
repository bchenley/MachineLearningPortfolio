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
