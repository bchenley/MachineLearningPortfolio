import re, string, contractions

def preprocess_text(texts, 
                    remove_html_tags = True,
                    lowercase_text = True,
                    remove_punctuation = True,
                    expand_contractions = True):
  
  preprocess_texts = []

  for text in texts:
    if remove_html_tags: 
      text = re.sub(r"<.*?>", "", text)
    if lowercase_text: 
      text = text.lower()
    if remove_punctuation: 
      text = text.translate(str.maketrans("", "", string.punctuation))
    if expand_contractions: 
      text = contractions.fix(text)

    preprocess_texts.append(text)
  
  return preprocess_texts
