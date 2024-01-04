def plot_roc(fpr, tpr,
             ax = None,
             lw = 1,
             include_diag = True):

  if ax is None:
    fig, ax = plt.subplots(figsize = (5, 5))

  if include_diag:
    ax.plot([0, 1], [0, 1], '--k', lw = 1., alpha = 0.5)

  ax.plot(fpr, tpr, lw = lw)
  ax.grid(True)
  ax.set_xlim([-0.01, 1.01])
  ax.set_ylim([-0.01, 1.01])
