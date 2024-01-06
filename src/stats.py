from scipy.stats import skew

def skew_test(df, numeric_features):
  skew_scores = skew(df[numeric_features], axis = 0)

  results = pd.DataFrame({'Skew': skew_scores}, index = numeric_features)

  return results
