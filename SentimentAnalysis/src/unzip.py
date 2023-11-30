def unzip(source, sink = None):
  
  sink = sink or os.path.dirname(source)

  with zipfile.ZipFile(source, 'r') as zip_ref:
    zip_ref.extractall(sink)
