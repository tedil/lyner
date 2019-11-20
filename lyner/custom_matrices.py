import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabaz_score


def fit_distribution(matrix, distribution=nbinom, x0=None):
    def fit_distr_parallel(distribution, x0, values):
        import multiprocessing
        from joblib import Parallel, delayed
        return Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(distribution.fit)(x0, v) for x0, v in zip(x0, values))

    def estimate_fit_ml(distribution, x0):
        return fit_distr_parallel(distribution, x0, matrix.values)

    if not x0:
        x0 = list(zip(np.median(matrix.values, axis=1), mad(matrix.values, axis=1)))
    return estimate_fit_ml(distribution=distribution, x0=x0)


def mad(matrix, axis=None, c=1):
    a = np.asarray(matrix)
    center = np.apply_over_axes(np.nanmedian, a, axis)
    return np.nanmedian((np.abs(a - center)) / c, axis=axis)


def _make_cluster_criterion(matrix: pd.DataFrame, by=['trend']):
    cluster_data = pd.DataFrame(index=matrix.index)
    if isinstance(by, list):
        for trait in by:
            if trait == 'trend':
                d = matrix.diff(axis=1).fillna(0)
                d = np.sign(np.round(d))
            elif trait == 'mean':
                d = np.nanmean(matrix, axis=1)
            elif trait == 'median':
                d = np.nanmedian(matrix, axis=1)
            elif trait == 'var':
                d = np.nanvar(matrix, axis=1)
            elif trait == 'mad':
                d = mad(matrix, axis=1)
            else:
                raise ValueError("Unsupported trait {trait}.")
            if trait != 'trend':
                cluster_data[trait] = d
            else:
                cluster_data = pd.concat([cluster_data, d], axis=1)
    elif isinstance(by, pd.DataFrame):
        cluster_data = by
    else:
        raise ValueError(f'Cannot cluster by {by}.')
    return cluster_data


def cluster_targets(matrix: pd.DataFrame, by=['trend'], score=calinski_harabaz_score, min_nclusters=2,
                    max_nclusters=20):
    # `by` may be any comma-separated combination of 'trend' / 'mean' / 'median' / 'var' / 'mad'
    # `by` may also be a list of columns
    # `by` may also be a pd.DataFrame with the same shape & index & columns as `matrix`
    if not by:
        by = matrix.columns
        cluster_data = matrix[by]
    else:
        cluster_data = _make_cluster_criterion(matrix, by)

    ac = AgglomerativeClustering(memory='/tmp', compute_full_tree=True)

    def get_labels(n_clusters):
        ac.set_params(n_clusters=n_clusters)
        return ac.fit(cluster_data).labels_

    labels = [get_labels(i) for i in range(min_nclusters, min(max_nclusters + 1, cluster_data.shape[0]))]
    scores = list(map(lambda l: score(cluster_data, l), labels))

    i = np.argmax(scores)
    labels = labels[i]
    data2 = matrix.copy()
    data2['cluster'] = labels
    cluster_indices = [data2[(data2[['cluster']] == label)['cluster']].index for label in sorted(np.unique(labels))]
    return cluster_indices, labels


def estimate_size_factors(matrix: pd.DataFrame):  # 'estimateSizeFactorsForMatrix' from deseq
    logmatrix = np.log(matrix.values)
    loggeomeans = np.mean(logmatrix, axis=1)
    l = logmatrix - loggeomeans[:, np.newaxis]
    return np.exp(np.median(np.nan_to_num(l), axis=0))


def deseq_normalize(matrix: pd.DataFrame):
    return matrix / estimate_size_factors(matrix)
