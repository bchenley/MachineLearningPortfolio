import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def describe(df, missing_values = []):

  df = df.copy()
  df_mode = df.mode(axis = 0).apply(lambda x: ', '.join(x.dropna().astype(str)), axis = 0)
  
  skew_fn = lambda x: stats.skew(x) if pd.api.types.is_numeric_dtype(x) else np.nan

  df_ = pd.concat([df_mode, df.apply(skew_fn) , df.dtypes, df.nunique()], axis = 1)
  df_.columns = ['mode', 'dkew', 'ftype', 'cardinality']
  
  df_ = pd.merge(df.describe().T, df_, how = 'right', left_index = True, right_index = True)

  missing_values = missing_values
  df_['missing'] = 0
  for var in df_.index:
    df_.loc[df_.index == var, 'missing'] = (pd.isna(df[var]) | df[var].map(lambda x: x in missing_values)).sum()
  
  df_['count'] = df.count()

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

  def sign(self, idx = None, axis = 0):

    return np.sign(self.data[idx]) if idx is not None else np.sign(self.data)

  def mean(self, axis=0):
      """
      Compute the mean of the data along the specified axis.

      Parameters:
      axis (int, optional): The axis along which to compute the mean (default is 0).

      Returns:
      numpy.ndarray: The mean values computed along the specified axis.
      """
      return self.data.mean(axis=axis)

  def median(self, axis=0):
      """
      Compute the median of the data along the specified axis.

      Parameters:
      axis (int, optional): The axis along which to compute the median (default is 0).

      Returns:
      numpy.ndarray: The median values computed along the specified axis.
      """
      return np.median(self.data, axis=axis)

  def mode(self, axis=0):
      """
      Compute the mode and count of the mode values in the data along the specified axis.

      Parameters:
      axis (int, optional): The axis along which to compute the mode (default is 0).

      Returns:
      numpy.ndarray: The mode values computed along the specified axis.
      numpy.ndarray: The count of occurrences of the mode values computed along the specified axis.
      """
      mode_, count_ = stats.mode(self.data, axis=axis)
      return mode_, count_

  def sdev(self, axis=0, ddof=0):
      """
      Compute the standard deviation of the data along the specified axis.

      Parameters:
      axis (int, optional): The axis along which to compute the standard deviation (default is 0).
      ddof (int, optional): The delta degrees of freedom used in the calculation (default is 0).

      Returns:
      numpy.ndarray: The standard deviation values computed along the specified axis.
      """
      return self.data.std(axis=axis, ddof=ddof)

  def iqr(self, axis=0, range=[25, 75]):
      """
      Compute the interquartile range (IQR) of the data along the specified axis.

      Parameters:
      axis (int, optional): The axis along which to compute the IQR (default is 0).
      range (list, optional): The percentiles used to define the IQR (default is [25, 75]).

      Returns:
      numpy.ndarray: The IQR values computed along the specified axis.
      """
      return stats.iqr(self.data, axis=axis, rng=range)

  def outliers(self, axis=0, range = [25, 75], scale=1.5):
      """
      Identify outliers in the data along the specified axis using the IQR method.

      Parameters:
      axis (int, optional): The axis along which to identify outliers (default is 0).
      range (list, optional): The percentiles used to define the IQR (default is [25, 75]).
      scale (float, optional): The scaling factor to adjust the IQR threshold (default is 1.5).

      Returns:
      numpy.ndarray: The values of outliers identified along the specified axis.
      tuple: A tuple containing the indices of outliers along the specified axis.
      """
      iqr = self.iqr(axis=axis, range=range)
      lower_threshold = np.quantile(self.data, q=range[0]/100., axis=axis) - scale * iqr
      upper_threshold = np.quantile(self.data, q=range[1]/100., axis=axis) + scale * iqr

      outliers = (self.data < lower_threshold) | (self.data > upper_threshold)

      outlier_values = self.data[outliers]
      outlier_idx = np.where(outliers)

      return outlier_values, outlier_idx

  def clip(self, axis=0, range = [25, 75], scale=1.5):

    _, outlier_idx = self.outliers(axis = axis, range = range, scale = scale)
    
    data_clipped = self.data.copy()
    
    data_clipped[outlier_idx[0]] = self.median(axis=axis) + self.sign(idx=outlier_idx[0], axis=axis) * self.iqr(axis=axis, range=range)

    return data_clipped


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
