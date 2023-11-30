from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocess_text import preprocess_text

class TextPreprocessor(BaseEstimator, TransformerMixin):
  def __init__(self, 
              remove_html_tags = True,
              lowercase_text = True,
              remove_punctuation = True,
              expand_contractions = True):

    self.remove_html_tags = remove_html_tags
    self.lowercase_text = lowercase_text
    self.remove_punctuation = remove_punctuation
    self.expand_contractions = expand_contractions
  
  def fit(self, texts, y = None):
    return self
  
  def transform(self, texts):
    return preprocess_text(texts,
                           remove_html_tags = self.remove_html_tags,
                           lowercase_text = self.lowercase_text,
                           remove_punctuation = self.remove_punctuation,
                           expand_contractions = self.expand_contractions)
