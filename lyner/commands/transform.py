import logging

import click
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from natsort import natsorted
from numba import njit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale

from lyner._main import rnax
from lyner.click_extras import pass_pipe, arggist, Pipe, DICT, LIST

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)


def diffs(_cluster, data, order):
    sorted_data = np.sort(data, kind='mergesort')
    n = data.shape[0]
    diffs = np.empty((n - order))

    @njit()
    def _foo(diffs, data, order):
        for i in range(n - order):
            diffs[i] = data[i + order] - data[i]

    _foo(diffs, sorted_data, order)
    return pd.Series(diffs)


@rnax.command()
@click.argument('type')
@pass_pipe
@arggist
def astype(pipe: Pipe, type: str):
    """Convert data to given type."""
    if hasattr(np, type):
        typecode = getattr(np, type, None)
        if typecode:
            pipe.matrix = pipe.matrix.astype(typecode)
        else:
            raise ValueError(f"Unknown type `{type}`")


@rnax.command()
@click.argument('value', type=click.FLOAT)
@pass_pipe
@arggist
def threshold(pipe: Pipe, value):
    """Set |data| < value to 0, data >= value to 1, -data >= value to -1."""
    data = getattr(pipe, pipe.selection, pipe.matrix)
    data.values[np.abs(data) < value] = 0
    data.values[data >= value] = 1
    data.values[-data >= value] = -1


@rnax.command()
@click.option('--method', '-m', type=click.Choice(['mean', 'median']), default='median')
@pass_pipe
@arggist
def center(pipe: Pipe, method: str):
    """Center features around their respective median or mean."""
    data = getattr(pipe, pipe.selection, pipe.matrix)
    if method == 'median':
        data.values[:] = data.subtract(np.nanmedian(data.values, axis=0))[:]
    elif method == 'mean':
        data.values[:] = data.subtract(np.nanmean(data.values, axis=0))[:]


@rnax.command()
@click.option("-m", "--method", type=click.Choice(['pearson', 'kendall', 'spearman']), default='pearson')
@pass_pipe
@arggist
def correlate(pipe: Pipe, method: str):
    """Correlate features using either of pearson, kendall or spearman correlation."""
    pipe.matrix = pipe.matrix.corr(method=method)


@rnax.command()
@click.option("--mode", "-m", type=click.Choice(['PCA', 'KPCA', 'NMF', 'BMF', 'TSNE', 'ICA']), default='PCA')
@click.option("--decode", "-d", is_flag=True)
@click.option("--num-components", "-n", type=click.INT, default=2)
@click.option("--mode-config", "-c", type=DICT, default={})
@pass_pipe
@arggist
def decompose(pipe: Pipe, mode: str, num_components: int, decode: bool, mode_config: dict):
    """Decomposition/dimensionality reduction (PCA, ICA, …)"""
    matrix = pipe.matrix.copy()
    mc = {k: v for k, v in mode_config.items()}
    if matrix.isnull().values.any():
        LOGGER.warning("Dropping rows containing nan values")
        matrix.dropna(how='any', inplace=True)
    data = matrix.values
    if mode == 'PCA':
        from sklearn.decomposition import PCA
        decomposition = PCA
    elif mode == 'ICA':
        from sklearn.decomposition import FastICA
        decomposition = FastICA
    elif mode == 'KPCA':
        from sklearn.decomposition import KernelPCA
        decomposition = KernelPCA
    elif mode == 'NMF':
        from sklearn.decomposition import NMF
        decomposition = NMF
    elif mode == 'BMF':
        raise NotImplementedError("BMF is not implemented yet.")
    elif mode == 'TSNE':
        from sklearn.manifold import TSNE
        decomposition = TSNE
    else:
        raise ValueError(f"Unknown mode {mode}.")  # can't happen because mode is a click.Choice

    mode_config.setdefault('n_components', num_components)
    LOGGER.info(f"Computing {mode} with {num_components} components.")
    decomposition = decomposition(**mode_config)
    X_r = decomposition.fit_transform(data.T)
    if decode:
        LOGGER.info(f"Computing {mode}⁻¹.")
        X_r = decomposition.inverse_transform(X_r)
        index = pipe.matrix.index
    else:
        index = [f"{mode}_{i}" for i in range(num_components)]
    labels = matrix.columns.values.tolist()
    pipe.decomposition = decomposition
    pipe._index = pipe.matrix.index
    pipe._columns = pipe.matrix.columns
    pipe.matrix = pd.DataFrame(data=X_r.T,
                               columns=labels,
                               index=index)

    mode_config.clear()
    mode_config.update(mc)


@rnax.command()
@click.option('--order', '-o', type=click.INT, default=1)
@pass_pipe
@arggist
def mmr(pipe: Pipe, order: int):
    """Calculate columnwise differences (of order `order`)"""
    df: pd.DataFrame = pipe.matrix
    groups, n = (df.groupby(level=0), len(df.index)) if pipe.is_clustered and df.index.nlevels > 1 \
        else (df.iterrows(), df.shape[0])

    data_subsets = Parallel(n_jobs=-1, verbose=1)(delayed(diffs)(cluster, data, order) for cluster, data in groups)
    pipe.matrix = pd.concat(data_subsets, axis=1).T
    pipe.matrix.index = df.index
    pipe.matrix.columns = [f"diff_{i:03d}" for i in range(len(df.columns) - order)]


@rnax.command()
@click.option('--sum', '-s', type=click.INT, default=None,
              help="Drops rows with sum smaller than or equal to given value.")
@click.option('--zeros', '-z', type=click.INT, default=None,
              help="Drop rows with up to the given amount of zeros.")
@click.option('--identical', '-i', is_flag=True, help="Drop rows consisting of only one single value.")
@click.option('--negative', '-n', is_flag=True, help="Drop rows with negative entries.")
@click.option('--drop-na', '-e', is_flag=True, help="Drop rows with NA/nan/empty entries.")
@click.option('--drop-duplicates', '-d', is_flag=True, help="Drop duplicate rows.")
@click.option('--prefix', '-p', type=LIST, default=[])
@click.option('--suffix', type=LIST, default=[])
@click.option('--variance-relative', '-v', type=click.FLOAT, default=1.,
              help="Keep the top n% most variant rows, drop the rest.")
@click.option('--variance-absolute', '-k', type=click.INT, default=0,
              help="Keep the top k most variant rows, drop the rest.")
@pass_pipe
@arggist
def filter(pipe: Pipe,
           sum: int,
           zeros: int,
           drop_na: bool,
           drop_duplicates: bool,
           identical: bool,
           negative: bool,
           variance_relative: float,
           variance_absolute: int,
           prefix: list,
           suffix: list):
    """Filter data according to selected option."""
    if drop_na:
        num_features = pipe.matrix.shape[0]
        pipe.matrix.dropna(how='any', inplace=True)
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows containing NAs. New dimensions: {pipe.matrix.shape}")
    if identical:
        num_features = pipe.matrix.shape[0]
        df: pd.DataFrame = pipe.matrix
        pipe.matrix = df.loc[~(df.nunique(axis=1) == 1)]
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} single-valued rows. New dimensions: {pipe.matrix.shape}")
    if drop_duplicates:
        num_features = pipe.matrix.shape[0]
        pipe.matrix.drop_duplicates(keep='first', inplace=True)
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} duplicate rows. New dimensions: {pipe.matrix.shape}")
    if prefix:
        num_features = pipe.matrix.shape[0]
        drop = set()
        for p in prefix:
            for row in pipe.matrix.index:
                if row.startswith(p):
                    drop.add(row)
        targets = set(list(pipe.matrix.index))
        drop = targets & drop
        pipe.matrix = pipe.matrix.drop(list(drop), axis=0)
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows prefixed with {prefix}. New dimensions: {pipe.matrix.shape}")
    if suffix:
        num_features = pipe.matrix.shape[0]
        drop = set()
        for p in suffix:
            for row in pipe.matrix.index:
                if row.endswith(p):
                    drop.add(row)
        targets = set(list(pipe.matrix.index))
        drop = targets & drop
        pipe.matrix = pipe.matrix.drop(list(drop), axis=0)
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows suffixed with {suffix}. New dimensions: {pipe.matrix.shape}")
    if negative:
        num_features = pipe.matrix.shape[0]
        pipe.matrix = pipe.matrix[~(pipe.matrix < 0.).any(axis=1)]
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows with negative entries. New dimensions: {pipe.matrix.shape}")
    if sum:
        num_features = pipe.matrix.shape[0]
        pipe.matrix = pipe.matrix.loc[pipe.matrix[pipe.matrix.sum(axis=1) > sum].index]
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows with sum <= {sum}. New dimensions: {pipe.matrix.shape}")
    if zeros:
        num_features = pipe.matrix.shape[0]
        threshold = pipe.matrix.shape[1] if zeros == -1 else zeros
        pipe.matrix = pipe.matrix.loc[pipe.matrix[(pipe.matrix != 0).sum(axis=1) >= threshold].index]
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} 0-rows. New dimensions: {pipe.matrix.shape}")
    if 0 < variance_absolute:
        num_features = pipe.matrix.shape[0]
        variance_relative = variance_absolute / pipe.matrix.shape[0]
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows with lowest variance. New dimensions: {pipe.matrix.shape}")
    if variance_relative < 1.:
        num_features = pipe.matrix.shape[0]
        matrix = pipe.matrix
        top = int(variance_relative * matrix.shape[0])
        var = np.nanvar(matrix.values, axis=1, ddof=1)
        # total_var = np.nansum(var)  # total sum of variance
        a = np.argsort(var, kind="mergesort")
        a = a[::-1]
        sel = np.zeros(matrix.shape[0], dtype=np.bool_)
        sel[a[:top]] = True
        # lost_p = matrix.shape[0] - top
        # lost_var = total_var - np.sum(var[sel])
        matrix = matrix.loc[sel]
        pipe.matrix = matrix
        num_dropped = num_features - pipe.matrix.shape[0]
        LOGGER.info(f"Dropped {num_dropped} rows with lowest variance. New dimensions: {pipe.matrix.shape}")


@rnax.command(aliases=['normalize'])
@click.argument('method', type=click.Choice(['quantile', 'deseq', 'identity', 'scale', 'unit', 'tanh']),
                default='quantile')
@click.option('--axis', '-a', type=click.IntRange(min=-1, max=1), default=0)
@pass_pipe
@arggist
def normalise(pipe: Pipe, axis: int, method: str):
    """Normalize data using one of the following methods: quantile, deseq, identity, scale, unit, tanh."""
    data = getattr(pipe, pipe.selection, pipe.matrix)
    if method == 'scale':
        if axis == -1:
            data.values[:] = scale(data.values.flatten()).reshape(data.shape)
        else:
            data.values[:] = scale(data.values, axis=axis)
    elif method == 'unit':
        v = data.values
        if axis == -1:
            data.values[:] = np.interp(v, (v.min(), v.max()), (0, +1))
        else:
            data2 = data.values
            data2 -= np.min(data2, axis=axis)
            data2 /= np.max(data2, axis=axis)
            data.values[:] = data2
    elif method == 'tanh':
        v = data.values
        if axis == -1:
            data.values[:] = np.interp(v, (v.min(), v.max()), (-1, +1))
        else:
            raise NotImplementedError()
    elif method == 'quantile':
        # based on https://stackoverflow.com/a/41078786
        data2 = getattr(pipe, pipe.selection, pipe.matrix).copy()
        rank_median = data2.stack().groupby(data2.rank(method='first').stack().astype(int)).median()
        data = data2.rank(method='min').stack().astype(int).map(rank_median).unstack()
        setattr(pipe, pipe.selection, data)
    else:
        data = data.normalize(method)
        setattr(pipe, pipe.selection, data)


@rnax.command()
@pass_pipe
@arggist
def sort(pipe: Pipe):
    """Sort values by columns."""
    pipe.matrix.sort_values(by=pipe.matrix.columns.values.tolist(), axis=0, inplace=True)


@rnax.command()
@pass_pipe
@arggist
def sort_index(pipe: Pipe):
    """Sort index."""
    pipe.matrix.sort_index(kind='mergesort', inplace=True)


@rnax.command()
@pass_pipe
@arggist
def reindex(pipe: Pipe):
    """Sort and reindex."""
    pipe.matrix = pipe.matrix.reindex(index=natsorted(pipe.matrix.index))


@rnax.command()
@click.argument('func', type=click.Choice(['log2', 'log10', 'log', 'exp', 'log1p', 'expm1', 'transpose']),
                default='log2')
@pass_pipe
@arggist
def transform(pipe: Pipe, func: str):
    """Apply a transformation to the current selection."""
    data = getattr(pipe, pipe.selection, pipe.matrix)
    if hasattr(np, func):
        func = getattr(np, func, 'log2')
    setattr(pipe, pipe.selection, func(data))


@rnax.command(aliases=['T'])
@pass_pipe
@arggist
def transpose(pipe: Pipe):
    """Transpose current selection if it is a matrix"""
    data = getattr(pipe, pipe.selection, pipe.matrix)
    setattr(pipe, pipe.selection, np.transpose(data))


@rnax.command()
@pass_pipe
@arggist
def compose(pipe: Pipe):
    """'Inverse' of `decompose`. Assumes `decompose` has been executed already."""
    assert hasattr(pipe, 'decomposition')
    assert hasattr(pipe, '_index')
    assert hasattr(pipe, '_columns')
    X_r = pipe.decomposition.inverse_transform(pipe.matrix.values.T)
    index = pipe._index.values.flatten()
    columns = pipe._columns.values.flatten()
    if X_r.shape == (len(index), len(columns)):
        X_r = X_r.T
    pipe.matrix = pd.DataFrame(data=X_r.T, index=index, columns=columns)


@rnax.command(aliases=['summarize'])
@click.argument('method', type=click.Choice(['median', 'mean', 'min', 'max']), default='mean')
@pass_pipe
@arggist
def summarise(pipe: Pipe, method: str):
    """Calculate either of median/mean/min/max for each group."""
    m: pd.DataFrame = pipe.matrix
    pipe.matrix = m.groupby(level=0, axis=1).transform(method)
