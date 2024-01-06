import pandas as pd
from sklearn.feature_selection import chi2, f_classif

def chi2_test(df, categorical_features, target):
  chi2_scores, p_values = chi2(df[categorical_features], df[target])

  results = pd.DataFrame({      
      'Chi2 Score': chi2_scores,
      'P-value': p_values
  }, index = categorical_features)

  return results

def anova_f_test(df, numeric_features, target):
  f_scores, p_values = f_classif(df[numeric_features], df[target])

  results = pd.DataFrame({      
      'F Score': f_scores,
      'P-value': p_values
  }, index = numeric_features)

  return results
