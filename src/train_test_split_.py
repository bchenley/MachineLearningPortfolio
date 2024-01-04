def train_test_split_(df: pd.DataFrame,
                      stratify_on = None,
                      train_size = 0.8,
                      random_state = None,
                      reset_index = False):

  df = df.copy()

  if stratify_on:
    for i,name in enumerate(stratify_on):
      if i == 0:
        df['stratify_on'] = df[name].astype(str)
      else:
        df['stratify_on'] += "_" + df[name].astype(str)

  df_ = df.drop(columns = ['stratify_on']) if stratify_on else df

  df_train, df_test = train_test_split(df_,
                                       train_size = train_size,
                                       stratify = df['stratify_on'] if stratify_on else None,
                                       random_state = random_state)

  if reset_index:
    df_train.reset_index(drop = True, inplace = True)
    df_test.reset_index(drop = True, inplace = True)

  return df_train, df_test
