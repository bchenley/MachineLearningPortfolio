import pandas as pd
import numpy as np

def describe(df, missing_values = []):

  df = df.copy()
  df_mode = df.mode(axis = 0).apply(lambda x: ', '.join(x.dropna().astype(str)), axis = 0)
  
  df_ = pd.concat([df_mode, df.dtypes, df.nunique()], axis = 1)
  df_.columns = ['Mode', 'Dtype', 'Cardinality']
  
  df_ = pd.merge(df.describe().T, df_, how = 'right', left_index = True, right_index = True)

  missing_values = [np.nan] + missing_values
  df_['missing'] = 0
  for var in df_.index:
    df_.loc[df_.index == var, 'missing'] = df[var].map(lambda x: x in missing_values).sum()
  
  df_['count'] = df.count()

  return df_
    
def annotate_bar(ax, show_height = True, show_percent = False):
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
        height = f"{p.get_height():.2f}"
        percent = f"({((p.get_height() / total) * 100):.1f}%)"

        if show_height and show_percent:
          note = f"{height} {percent}"
        elif show_height:
          note = height
        elif show_percent:
          note = percent

        # Add the annotation to the bar at the specified coordinates
        ax.annotate(note, (x, y), ha='center', va='bottom')
