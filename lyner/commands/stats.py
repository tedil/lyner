import logging
from itertools import zip_longest

import click
import numpy as np
import scipy.stats
from pandas import MultiIndex

from lyner import custom_distributions
from lyner._main import rnax
from lyner.click_extras import pass_pipe, arggist, Pipe
from lyner.custom_distributions import fit_distribution

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)


@rnax.command()
@click.option('--distribution', '-d',
              type=click.STRING, default='t',
              help="May be any of ['negbinom', 'gamma', 'laisson', 't', 'norm', 'cauchy', 'lognorm']"
                   " as well as any distribution in `scipy.stats.rv_continuous`.")
@pass_pipe
@arggist
def estimate(pipe: Pipe, distribution: str):
    """Fit the given distribution to each target(-cluster) and each (design-)group."""
    matrix = pipe.matrix
    distr = getattr(custom_distributions, distribution, None)
    if distr is None:
        distr = getattr(scipy.stats, distribution)
        if not isinstance(distr, scipy.stats.rv_continuous):
            raise TypeError(
                f'Only continuous variables are supported at the moment (negative binomial being the only exception).'
                f'\n{distr} is of type {type(distr)}.')
    pipe.distribution = distr

    def grouper(iterable, n, fillvalue=None):  # python docs recipe.
        """Collect data into fixed-length chunks or blocks
        grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"""
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)

    # step 1: if genes haven't been clustered, have each single gene be a singleton cluster
    if not isinstance(matrix.index, MultiIndex):
        index = matrix.index
        tuples = list(enumerate(list(index)))
        index2 = MultiIndex.from_tuples(tuples)
        matrix.index = index2
        matrix.index.names = ['Cluster', 'Gene']

    # step 2: if no samples have been grouped (via `design`), make sure there's at least one SampleGroup 'all'
    if not isinstance(matrix.columns, MultiIndex):
        columns = matrix.columns
        tuples = [('all', sample) for sample in list(columns)]
        columns2 = MultiIndex.from_tuples(tuples)
        matrix.columns = columns2
        matrix.columns.names = ['SampleGroup', 'Sample']

    all_sample_group_ids = matrix.columns.get_level_values('SampleGroup').unique()  # which classes/groups are there?
    all_cluster_ids = matrix.index.get_level_values('Cluster').unique()  # which clusters are there?
    num_samples = len(matrix.columns.get_level_values('Sample'))

    # the following lines of code are especially ugly. sorry.
    values = []
    cs, ss = [], []  # TODO: let cs be an ordered set
    for cluster_id, cluster in matrix.groupby(level=0, axis=0):  # for each cluster
        for sample in grouper(cluster.groupby(level=0, axis=1),
                              len(all_sample_group_ids)):  # for each sample in that cluster
            all_sample_ids = []
            all_sample_values = []
            for sample_group_id, sample_group in sample:
                # make sure all arrays have the same length, so we can pass in a 2d numpy array to fit_distribution
                v = sample_group.values.ravel()
                n = num_samples * cluster.values.shape[0] - v.shape[0]
                v = np.append(v, np.zeros(n) + np.nan)  # fill with nans (fit_distribution ignores these)

                values.append(v)
                cs.append(cluster_id)
                ss.append(sample_group_id)
                all_sample_ids.append(sample_group_id)
                all_sample_values.append(sample_group.values.ravel())
            values.append(np.array([a for l in all_sample_values for a in l]))
            cs.append(cluster_id)
            ss.append('::'.join(all_sample_ids))
    values = np.array(values)
    params = fit_distribution(values, distr)
    # sk.stats treats shape == None as loc & scale
    param_names = ['loc', 'scale'] if distr.shapes is None else distr.shapes.split(',')
    num_params = len(next(iter(params)))
    if distr.numargs < num_params:  # *sigh*
        if num_params - distr.numargs >= 1:
            param_names += ['loc']
        if num_params - distr.numargs >= 2:
            param_names += ['scale']
    row_index = pd.Index(np.unique(cs))
    column_index = pd.MultiIndex.from_tuples(list(zip(ss, param_names)))
    pipe.estimate = pd.DataFrame(index=row_index, columns=column_index, dtype=float)
    for c, s, est in zip(cs, ss, params):
        for pname, p in zip(param_names, est):
            pipe.estimate.loc[c, (s, pname)] = p
    pipe.estimate = pipe.estimate.sort_index(axis=1)
    pipe.estimate.index.names = ['Cluster']