import pandas as pd

def describe(df):
  df_ = pd.concat([df.dtypes, df.nunique()], axis = 1)
  df_.columns = ['Dtype', 'Cardinality']
  
  df_['Unknown'] = 0
  for col in df_.index:
    df_.loc[df_.index == col, 'Unknown'] = (df[col] == 'unknown').sum()

  return df_
