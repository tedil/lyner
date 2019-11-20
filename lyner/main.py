from pandas import IndexSlice
from scipy.optimize import fsolve

from .commands.plot import *
from .commands.transform import *
from .commands.stats import *
from .commands.keras import *
from .commands.cluster import *
from .commands.io import *
from .commands.bio import *

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)

# used for slicing the multiindex
# e.g. data.loc[idx[:, gene_cluster], idx[:, sample]] or data.loc[idx[:, gene_cluster], samplegroup]
idx = IndexSlice


@rnax.command()
@click.argument('seed')
@pass_pipe
@arggist
def seed(seed: int):
    """Sets both numpy and tensorflow seed."""
    from tensorflow import set_random_seed
    np.random.seed(seed)
    set_random_seed(seed)


@rnax.command()
@click.option('--mode', '-m', type=click.Choice(['likelihood', 'cdf']), default='likelihood')
@pass_pipe
@arggist
def changes(pipe: Pipe, mode: str):
    """Calculate differences between sample groups."""
    matrix = pipe.matrix
    if 'estimate' not in pipe:
        raise ValueError('Did not find any estimates. Forgot to call `estimate` first?')
    estimates = pipe.estimate  # rows: clusters/targets, columns: samplegroups
    chngs = pipe.changes = pd.DataFrame(index=matrix.index, columns=[mode])
    distr = pipe.distribution
    sample_group_ids = estimates.columns.get_level_values(0)
    for cluster_id, data in estimates.iterrows():
        pabc = {sid: data[sid] for sid in sample_group_ids}
        if len(pabc) != 3:
            raise ValueError('More than two groups not supported yet.')

        pc = [(k, p) for k, p in pabc.items() if '::' in k][0]
        pckey, pc = pc
        pab = pabc
        del pab[pckey]
        pa, pb = list(pab.items())
        pakey, pa = pa
        pbkey, pb = pb

        if mode == 'likelihood':
            llh = {}
            for sid in sample_group_ids:
                select = sid.split('::')
                values = matrix.loc[cluster_id, select].values.ravel()
                llh[sid] = -np.mean(np.log(distr.pdf(values, *data[sid])))
            chngs.loc[cluster_id, mode] = llh[pckey] - (llh[pakey] + llh[pbkey]) / 2
        elif mode == 'cdf':
            values = matrix.loc[cluster_id, pckey.split('::')].values.ravel()
            minv = np.min(values)
            maxv = np.max(values)
            x = np.linspace(minv, maxv, 100)
            if np.argmax(distr.pdf(x, *pa)) > np.argmax(
                    distr.pdf(x, *pb)):  # such that pa's max is always to the left of pb's
                pa, pb = pb, pa

            # find x value(s) of intersection point(s)
            intersection = fsolve(lambda x: distr.pdf(x, *pa) - distr.pdf(x, *pb), x)

            # cut off solutions found to the left of smallest x in searchspace
            intersection = intersection[intersection > minv]

            # cut off solutions found to the right of largest x in searchspace
            intersection = np.unique(np.round(intersection[intersection < maxv], 4))

            if len(intersection) == 0:  # no intersection at all
                area_ab = 0
            else:
                smaller = np.minimum(distr.pdf(x, *pa), distr.pdf(x, *pb))
                # the larger the intersecting area, the less distinct the two groups (from each other)
                area_ab = np.trapz(smaller, x)
            area_c = np.trapz(distr.pdf(x, *pc), x)
            chngs.loc[cluster_id, mode] = (area_c - area_ab) / area_c
    pipe.changes = chngs.sort_values(by=[mode])
    pipe.matrix = pipe.matrix.reindex(pipe.changes.index)


@rnax.command()
@click.argument('what')
@pass_pipe
@arggist
def select(pipe: Pipe, what: str):
    """Select a datum based on its name (e.g. 'matrix' or 'estimate'), making it the target of commands such as
    `show`, `save` and `plot`.
    """
    pipe.selection = what


def main():
    # enforce at least one call to `estimate` before any call to `changes`
    if 'changes' in sys.argv and 'estimate' in sys.argv:
        assert (sys.argv.index('estimate') < sys.argv.index('changes'))
    rnax()
