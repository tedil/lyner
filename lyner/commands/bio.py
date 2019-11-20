import gzip
import logging

import click
import pandas as pd

from lyner._main import rnax
from lyner.click_extras import pass_pipe, arggist, Pipe

LOGGER = logging.getLogger("lyner")


@rnax.command()
@click.argument('file', type=click.Path())
@pass_pipe
@arggist
def read_annotation(pipe: Pipe, file: str):
    """Reads annotation from given file and stores it in `annotation`."""
    opener = gzip.open if file.endswith('.gz') else open
    with opener(file) as r:
        d = pd.read_csv(r, sep='\t', comment='#', index_col=0)
    d = d.drop(columns=[v for v in list(d.columns) if v.startswith("%")])
    pipe.annotation = d
