import logging

import click
from click_aliases import ClickAliasedGroup

from lyner.click_extras import pass_pipe, arggist, Pipe

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)


@click.group(chain=True, cls=ClickAliasedGroup)
@click.option("-v", "--verbose", count=True, default=0)
@pass_pipe
@arggist
def rnax(pipe: Pipe, verbose: int):
    """Sandbox and toolkit for RNASeq analysis.
    Commands can be chained arbitrarily."""
    LOGGER.setLevel(level=max(0, (5 - verbose) * 10))
    pipe.command = []
