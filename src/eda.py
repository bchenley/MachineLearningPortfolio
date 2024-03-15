import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def describe(df, missing_values = []):

  df = df.copy()
  
  dfd_T = df.describe(include = 'all').T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
  
  df_mode = df.mode(axis = 0).median()
  # mode_fn = lambda x: stats.mode(x)[0]

  skew_fn = lambda x: stats.skew(x) if pd.api.types.is_numeric_dtype(x) and x.nunique() > 1 else np.nan

  df_ = pd.concat([df_mode, df.apply(skew_fn) , df.dtypes, df.nunique()], axis = 1)
  df_.columns = ['mode', 'skew', 'dtype', 'cardinality']

  df_ = pd.merge(dfd_T, df_, how = 'right', left_index = True, right_index = True)

  missing_values = missing_values
  df_['missing'] = 0
  for var in df_.index:
    df_.loc[df_.index == var, 'missing'] = (pd.isna(df[var]) | df[var].map(lambda x: x in missing_values)).sum()

  df_['count'] = df_['count'].astype(int)
  df_['missing'] = df_['missing'].astype(int)
  df_['cardinality'] = df_['cardinality'].astype(int)

  return df_
    
def annotate_bar(ax, 
                 show_height = True, show_percent = False, 
                 decimals = 2, fontsize = 12):
    """
    Display the count and optional percentage on top of each bar in a bar plot.

    Parameters:
    ax (matplotlib Axes): The Axes object representing the bar plot.
    with_pct (bool): If True, display the percentage alongside the count. Default is True.
    """

    # Calculate the total count of data points in the plot
    total = sum([p.get_height() for p in ax.patches])

    # Iterate through each bar patch in the plot
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2  # X-coordinate for annotation
        y = p.get_height()  # Height of the bar

        # Create the annotation text with count
        height = f"{p.get_height():.{decimals}f}"
        percent = f"({((p.get_height() / total) * 100):.1f}%)"

        if show_height and show_percent:
          note = f"{height} {percent}"
        elif show_height:
          note = height
        elif show_percent:
          note = percent

        # Add the annotation to the bar at the specified coordinates
        ax.annotate(note, (x, y), ha='center', va='bottom', fontsize = fontsize)

class Descriptor:
  """
  A class for computing descriptive statistics and identifying outliers in data.

  Parameters:
  data (numpy.ndarray): The input data for which descriptive statistics and outliers will be computed.
  """

  def __init__(self, data):
      """
      Initialize the Descriptor object with the input data.

      Parameters:
      data (numpy.ndarray): The input data for which descriptive statistics and outliers will be computed.
      """
      self.data = data

      self.median_ = None
      self.mean_ = None
      self.mode_ = None
      self.count_ = None
      self.sdev_ = None
      self.skew_ = None
      self.kurt_ = None
      self.iqr_ = None

  def sign(self, idx = None, axis = 0):

    self.sign_ = np.sign(self.data[idx]) if idx is not None else np.sign(self.data)

    return self.sign_

  def skew(self, axis=0):
    """
    Compute the skew of the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the skew (default is 0).

    Returns:
    numpy.ndarray: The skew values computed
    """
    self.skew_ = self.data.skew(axis=axis)

    return self.skew_

  def kurtosis(self, axis=0):
    """
    Compute the kurtosis of the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the kurtosis (default is 0).

    Returns:
    numpy.ndarray: The kurtosis values computed
    """
    self.kurt_ = self.data.kurtosis(axis=axis)

    return self.kurt_

  def mean(self, axis=0):
    """
    Compute the mean of the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the mean (default is 0).

    Returns:
    numpy.ndarray: The mean values computed along the specified axis.
    """
    
    self.mean_ = self.data.mean(axis=axis)

    return self.mean_

  def median(self, axis=0):
    """
    Compute the median of the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the median (default is 0).

    Returns:
    numpy.ndarray: The median values computed along the specified axis.
    """
    self.median_ = np.median(self.data, axis=axis)

    return self.median_

  def mode(self, axis=0):
    """
    Compute the mode and count of the mode values in the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the mode (default is 0).

    Returns:
    numpy.ndarray: The mode values computed along the specified axis.
    numpy.ndarray: The count of occurrences of the mode values computed along the specified axis.
    """
    self.mode_, self.count_ = stats.mode(self.data, axis=axis)

    return self.mode_, self.count_

  def sdev(self, axis=0, ddof=0):
    """
    Compute the standard deviation of the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the standard deviation (default is 0).
    ddof (int, optional): The delta degrees of freedom used in the calculation (default is 0).

    Returns:
    numpy.ndarray: The standard deviation values computed along the specified axis.
    """
    self.sdev_ = self.data.std(axis=axis, ddof=ddof)

    return self.sdev_

  def iqr(self, axis=0, range=[.25, .75]):
    """
    Compute the interquartile range (IQR) of the data along the specified axis.

    Parameters:
    axis (int, optional): The axis along which to compute the IQR (default is 0).
    range (list, optional): The percentiles used to define the IQR (default is [.25, .75]).

    Returns:
    numpy.ndarray: The IQR values computed along the specified axis.
    """
    self.iqr_ = stats.iqr(self.data, axis=axis, rng=range)

    return self.iqr_

  def outliers(self, axis=0, range = [.25, .75], scale=1.5):
    """
    Identify outliers in the data along the specified axis using the IQR method.

    Parameters:
    axis (int, optional): The axis along which to identify outliers (default is 0).
    range (list, optional): The percentiles used to define the IQR (default is [.25, .75]).
    scale (float, optional): The scaling factor to adjust the IQR threshold (default is 1.5).

    Returns:
    numpy.ndarray: The values of outliers identified along the specified axis.
    tuple: A tuple containing the indices of outliers along the specified axis.
    """
    iqr = self.iqr(axis=axis, range=range)
    lower_threshold = np.quantile(self.data, q=range[0], axis=axis) - scale * iqr
    upper_threshold = np.quantile(self.data, q=range[1], axis=axis) + scale * iqr

    outliers = (self.data < lower_threshold) | (self.data > upper_threshold)

    outlier_values = self.data[outliers]
    outlier_idx = np.where(outliers)
    
    return outlier_values, outlier_idx

  def plot_outliers(self, axis = 0, range = [.25, .75], scale = 1.5, ax = None):

    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    
    outlier_values, outlier_idx = self.outliers(axis = axis, range = range, scale = scale)  
    
    ax.plot(self.data.values if isinstance(self.data, pd.Series) else self.data, '.k', 
            label = self.data.name if isinstance(self.data, pd.Series) else None)    
    ax.plot(outlier_idx[0], outlier_values, 'o', markerfacecolor = 'none', markeredgecolor = 'r', markersize = 5,
            label = 'Outliers')
    ax.set_title(f"Outliers in {self.data.name}" if isinstance(self.data, pd.Series) else f"Outliers")
    ax.set_xlabel('Sample Index')
    ax.set_ylabel(self.data.name if isinstance(self.data, pd.Series) else None)
    ax.grid()
    ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.95))

  def treat_outliers(self, method = 'clip', axis=0, range = [.25, .75], scale=1.5):

    _, outlier_idx = self.outliers(axis = axis, range = range, scale = scale)
    
    data_treated = self.data.copy()
    
    if self.median_ is not None:
      median_ = self.median_ 
    else:
      median_ = self.median(axis = axis)

    if method == 'clip':
      data_treated[outlier_idx[0]] = median_ + self.sign(idx=outlier_idx[0], axis=axis) * self.iqr(axis=axis, range=range)
    elif method == 'median':
      data_treated[outlier_idx[0]] = median_
    elif method == 'mean':
      if self.mean_ is not None:
        mean_ = self.mean_ 
      else:
        mean_ = self.mean(axis = axis)

      data_treated[outlier_idx[0]] = mean_

    return data_treated


def plot_outliers(df, axis=0, range=[25, 75], scale=1.5, 
                    fig_num = None, figsize = None):

  numeric_cols = df.select_dtypes(include = ['number']).columns.to_list()
  
  # Count the number of columns with outliers
  num_outliers = sum([len(Descriptor(df[col]).outliers(axis=axis, range=range, scale=scale)[0]) > 0 for col in numeric_cols])

  # Calculate the grid size for subplots
  nrows = ncols = int(np.ceil(np.sqrt(num_outliers)))

  # Set default figsize if not provided
  if figsize is None:
      figsize = (5 * ncols, 5 * nrows)

  fig, ax = plt.subplots(nrows, ncols, figsize=figsize, num=fig_num)

  x = np.arange(df.shape[0])

  axf = ax.flatten()
  j = -1
  for i,col in enumerate(numeric_cols):

    descriptor = Descriptor(df[col])
    
    outliers, outlier_x = descriptor.outliers(axis = axis, 
                                              range = range, 
                                              scale = scale)

    if len(outliers) > 0:      
      j += 1      
      axf[j].plot(x, df[col].values, 'k', label = 'Data')
      axf[j].plot(outlier_x[0], outliers, '.r', label = 'Outliers')
      axf[j].set_title(f"Outliers in {col}")
      axf[j].set_xlabel("Index")
      axf[j].legend()

  plt.tight_layout()
  plt.show()
