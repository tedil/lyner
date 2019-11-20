import logging
import sys
from typing import Tuple

import click
import pandas as pd
from pandas import DataFrame, MultiIndex

from lyner._main import rnax
from lyner.click_extras import LIST, pass_pipe, arggist, Pipe, DataFrameType

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)
idx = pd.IndexSlice


@rnax.command()
@click.argument('matrix', type=DataFrameType(exists=True, dir_okay=False, file_okay=True, drop_non_numeric=True))
@pass_pipe
def read(pipe: Pipe, matrix: Tuple[DataFrame, DataFrame]):
    """Read abundance/count matrix from `MATRIX` (tsv format).
    """
    LOGGER.info("Reading input file…")
    matrix, aux = matrix
    pipe.matrix = matrix.sort_index(kind='mergesort')  # sort genes lexicographically → enables slicing
    pipe.supplement = aux.sort_index(kind='mergesort') if aux is not None else None
    pipe.is_clustered = False
    pipe.is_grouped = False
    pipe.selection = 'matrix'
    LOGGER.info(f"Finished reading input file.\n"
                f"  Matrix dimensions: {matrix.shape[0]}x{matrix.shape[1]} (features x samples).")


@rnax.command()
@pass_pipe
@arggist
def show(pipe: Pipe):
    """Prints current selection to stdout, in tsv format."""
    # TODO allow storing plain matrix, i.e. without multiindex
    data = getattr(pipe, pipe.selection, pipe.matrix)
    data.to_csv(sys.stdout, sep='\t')


@rnax.command()
@click.argument('out', type=click.Path(writable=True, dir_okay=False, file_okay=True), default='-')
@click.option("--mode", "-m", type=click.Choice(['csv', 'pickle', 'auto']), default='auto')
@pass_pipe
@arggist
def store(pipe: Pipe, out: str, mode: str):
    """Save current selection in given file; in tsv format."""
    # TODO allow storing plain matrix, i.e. without multiindex
    data = getattr(pipe, pipe.selection, pipe.matrix)
    compression = 'infer'
    if mode == 'auto':
        if out.endswith('.gz'):
            f = out.rstrip('.gz')
            compression = 'gzip'
        else:
            f = out
        if f.endswith('.csv') or f.endswith('.tsv'):
            mode = 'csv'
        elif f.endswith('.pickle'):
            mode = 'pickle'
    if mode == 'csv':
        if compression == 'infer':
            compression = None
        data.to_dense().to_csv(out, sep="\t", index=True, index_label="Feature", compression=compression)
    elif mode == 'pickle':
        data.to_pickle(out, compression=compression)


@rnax.command()
@click.argument('supplementary-data', type=click.Path())
@pass_pipe
@arggist
def supplement(pipe: Pipe, supplementary_data: str):
    """Supply additional data which may be used for plot colors, for example."""
    d = pd.read_csv(supplementary_data, sep='\t', header=0, index_col=0, comment='#')
    pipe.supplement = d
    LOGGER.info(f"Supplementary information loaded: {d.columns.tolist()}")


@rnax.command()
@click.option('--targets', '-t', type=LIST)
@click.option('--from-file', '-f', type=click.File())
@click.option('--mode', '-m', type=click.Choice(['exclude', 'intersect']), default='intersect')
@pass_pipe
@arggist
def targets(pipe: Pipe, targets: list, from_file: click.File, mode: str):
    """Include only/exclude all genes in the given file. One feature per line."""
    if targets:
        targets = pd.DataFrame(data=targets, index=targets)
    elif from_file:
        targets = set([line.strip() for line in from_file])
        targets = pd.DataFrame(index=targets)
    else:
        raise ValueError("No targets specified.")

    if mode == 'intersect':
        g = sorted(list(targets.index))
    elif mode == 'exclude':
        g = sorted(list(set(pipe.matrix.index) - set(targets.index)))
    if isinstance(pipe.matrix.index, MultiIndex):
        pipe.matrix = pipe.matrix.loc[idx[:, g], :]
    else:
        drop_targets = sorted(list(set(pipe.matrix.index) - set(targets.index)))
        pipe.matrix = pipe.matrix.drop(drop_targets)


@rnax.command()
@click.argument('design', type=DataFrameType(exists=True, dir_okay=False, file_okay=True, drop_non_numeric=False))
@pass_pipe
@arggist
def design(pipe: Pipe, design: pd.DataFrame):
    """Description of the experiment. Expects 2-column tsv (Sample, Class)."""
    pipe.matrix = (pipe.matrix[design.index])

    # make sure index == lex_sort(index) to enable MultiIndex slicing
    # (assuming order of design.index is important)
    if sorted(list(design.index)) != list(design.index):
        reindexed = [f"{chr(33 + i)}__{idx}" for i, idx in enumerate(list(design.index))]
        design.index, _ = design.index.reindex(reindexed)
        pipe.matrix.columns = reindexed
    pipe.design = design

    # assume the first column is the only column and contains class labels
    # while the index column corresponds to sample ids
    cindex = pipe.design.index.groupby(pipe.design[pipe.design.columns[0]])
    tuples = [(samplegroup, sample) for samplegroup, sample_index in cindex.items() for sample in list(sample_index)]
    pipe.matrix.columns = MultiIndex.from_tuples(tuples, names=['SampleGroup', 'Sample'])
    pipe.matrix = pipe.matrix[sorted(pipe.matrix.columns)]
    pipe.is_grouped = True
    # pipe.groups = {sample_group: (list(pipe.matrix.loc[:, sample_group]), pipe.matrix.loc[:, sample_group].shape[1]) for sample_group in cindex.keys()}
