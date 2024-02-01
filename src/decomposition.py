from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def eig(data):
  eval, evec = np.linalg.eig(data)

  sort_idx = eval.argsort()[::-1]

  return eval[sort_idx], evec[:, sort_idx]
  
class PrincipalComponentAnalysis():
  def __init__(self,
               n_components = None, copy=True, whiten=False, svd_solver='auto',
               tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto',
               random_state=None):

    self.n_components = n_components
    
    self.pca = PCA(n_components = n_components,
                   copy = copy, whiten = whiten, svd_solver = svd_solver,
                   tol = tol, iterated_power = iterated_power, n_oversamples = n_oversamples,
                   power_iteration_normalizer = power_iteration_normalizer, random_state = random_state)
    
  def fit(self, data, kaiser = False, sval_threshold = 1.0, var_threshold = 1.0, digits = 4):

    # Fit
    self.pca.fit(data)

    # Get PCs and singular values
    pc, sval = self.pca.components_, self.pca.singular_values_

    # Extract the eigan values from the PCA singular values (λ = σ^2/[N-1])
    eval_cov = (sval**2)/(data.shape[0]-1)

    # Compute percent cumalative contribution
    sval_pct_cumsum = sval.cumsum()/sval.sum()

    # Compute percent cumaltive variance explained 
    var_pct_cumsum = self.pca.explained_variance_ratio_.cumsum()/self.pca.explained_variance_ratio_.sum()

    # Apply sval_threshold
    
    sval_mask = np.round(sval_pct_cumsum, digits) <= np.round(sval_threshold, digits)

    pc = pc[sval_mask, :]
    sval = sval[sval_mask]
    eval_cov = eval_cov[sval_mask]
    sval_pct_cumsum = sval_pct_cumsum[sval_mask]
    var_pct_cumsum = var_pct_cumsum[sval_mask]

    # Apply kaiser
    if kaiser:
      kaiser_mask = np.round(eval_cov,digits) >= 1.
      pc = pc[kaiser_mask, :]
      sval = sval[kaiser_mask]
      eval_cov = eval_cov[kaiser_mask]
      sval_pct_cumsum = sval_pct_cumsum[kaiser_mask]
      var_pct_cumsum = var_pct_cumsum[kaiser_mask]

    # If n_components is a float    
    var_mask = np.round(var_pct_cumsum, digits) <= np.round(var_threshold, digits)
    
    pc = pc[var_mask, :]
    sval = sval[var_mask]
    eval_cov = eval_cov[var_mask]
    sval_pct_cumsum = sval_pct_cumsum[var_mask]
    var_pct_cumsum = var_pct_cumsum[var_mask]

    self.pc, self.sval, self.eval_cov = pc, sval, eval_cov
    self.sval_pct_cumsum = sval_pct_cumsum
    self.var_pct_cumsum = var_pct_cumsum

    self.n_components = len(self.sval)
    
  def transform(self, data):

    return data.dot(self.pc.T)

  def fit_transform(self, data, kaiser = False, sval_threshold = 1.0, var_threshold = 1.0):

    self.fit(data, kaiser = kaiser, sval_threshold = sval_threshold, var_threshold = var_threshold)

    return self.transform(data)

  def plot(self, ax = None):

    if ax is None:
      fig, ax = plt.subplots(1, 3, figsize = (15, 5))

    ax[0].stem(self.eval_cov, basefmt = " ")
    ax[0].yaxis.grid(True)
    ax[0].set_ylabel('Eigen Values of Covariance Matrix')
    ax[0].set_xticks(np.arange(0, self.n_components))
    ax[0].set_xticklabels(np.arange(1, self.n_components+1))
    ax[0].set_xlabel('Eigen Value Order')
  
    ax[1].stem(self.sval_pct_cumsum*100, basefmt = " ")
    ax[1].yaxis.grid(True)
    ax[1].set_ylim([0, 101])
    ax[1].set_yticks(np.arange(0, 110, 10))
    ax[1].set_ylabel('% Cumalative Contribution of Singular Values')  
    ax[1].set_xticks(np.arange(0, self.n_components))
    ax[1].set_xticklabels(np.arange(1, self.n_components+1))
    ax[1].set_xlabel('Principal Component')

    ax[2].stem(self.var_pct_cumsum*100, basefmt = " ")
    ax[2].yaxis.grid(True)
    ax[2].set_ylim([0, 101])
    ax[2].set_yticks(np.arange(0, 110, 10))
    ax[2].set_ylabel('% Cumalative Variance Explained')  
    ax[2].set_xticks(np.arange(0, self.n_components))
    ax[2].set_xticklabels(np.arange(1, self.n_components+1))
    ax[2].set_xlabel('Principal Component')
