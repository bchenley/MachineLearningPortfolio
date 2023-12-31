from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class PrincipalComponentAnalysis():
  def __init__(self,
               n_components=None, copy=True, whiten=False, svd_solver='auto',
               tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto',
               random_state=None):

    self.pca = PCA(n_components = n_components,
                   copy = copy, whiten = whiten, svd_solver = svd_solver,
                   tol = tol, iterated_power = iterated_power, n_oversamples = n_oversamples,
                   power_iteration_normalizer = power_iteration_normalizer, random_state = random_state)

  def fit(self, data, kaiser = False, threshold = 1.):

    # Fit
    self.pca.fit(data)

    # Get PCs and singular values
    pc, sval = self.pca.components_, self.pca.singular_values_

    # Extract the eigan values from the PCA singular values (λ = σ^2/[N-1])
    eval_cov = (sval**2)/(data.shape[0]-1)

    # Compute percent cumalative contribution
    pct_cumsum = sval.cumsum()/sval.sum()

    # Apply threshold
    thresh_mask = pct_cumsum <= threshold
    pc = pc[thresh_mask, :]
    sval = sval[thresh_mask]
    eval_cov = eval_cov[thresh_mask]
    pct_cumsum = pct_cumsum[thresh_mask]

    # Apply kaiser
    if kaiser:
      kaiser_mask = eval_cov >= 1.
      pc = pc[kaiser_mask, :]
      sval = sval[kaiser_mask]
      eval_cov = eval_cov[kaiser_mask]
      pct_cumsum = pct_cumsum[kaiser_mask]

    self.pc, self.sval, self.eval_cov = pc, sval, eval_cov
    self.pct_cumsum = pct_cumsum

    self.n_components = len(self.sval)

  def transform(self, data):

    return data.dot(self.pc.T)

  def fit_transform(self, data, kaiser = False, threshold = 1.):

    self.fit(data, kaiser = kaiser, threshold = threshold)

    return self.transform(data)

  def plot(self, ax = None):

    if ax is None:
      fig, ax = plt.subplots(1, 2, figsize = (10, 5))

    ax[0].stem(self.eval_cov, basefmt = " ")
    ax[0].yaxis.grid(True)
    ax[0].set_ylabel('Eigen Values of Covariance Matrix')
    ax[0].set_xticks(np.arange(0, self.n_components))
    ax[0].set_xticklabels(np.arange(1, self.n_components+1))
    ax[0].set_xlabel('Eigen Value Order')
  
    ax[1].stem(self.pct_cumsum, basefmt = " ")
    ax[1].yaxis.grid(True)
    ax[1].set_ylim([0, 1.01])
    ax[1].set_yticks(np.arange(0, 1.05, .1))
    ax[1].set_ylabel('% Cumalative Contribution of Singular Values')  
    ax[1].set_xticks(np.arange(0, self.n_components))
    ax[1].set_xticklabels(np.arange(1, self.n_components+1))
    ax[1].set_xlabel('Principal Component')
