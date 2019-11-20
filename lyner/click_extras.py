import sys
from collections import Collection, namedtuple
from typing import Iterable

import click
import numpy as np
import pandas as pd
from click import Option, UsageError


# taken from https://stackoverflow.com/questions/37310718/mutually-exclusive-option-groups-in-python-click
class MutexOption(Option):
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop('mutually_exclusive', []))
        help = kwargs.get('help', '')
        if self.mutually_exclusive:
            ex_str = ', '.join(self.mutually_exclusive)
            kwargs['help'] = help + (
                    ' NOTE: This option is mutually exclusive with:'
                    ' [' + ex_str + '].'
            )
        super(MutexOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with:"
                " `{}`.".format(
                    self.name,
                    ', '.join(self.mutually_exclusive)
                )
            )

        return super(MutexOption, self).handle_parse_result(
            ctx,
            opts,
            args
        )


class ListParamType(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            if ',' in value:
                return value.split(',')
            else:
                return [value]
        else:
            if isinstance(value, list):
                return value
            else:
                return [value]


LIST = ListParamType()


class DictParamType(click.ParamType):
    name = 'dict'

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            def guess_type(t):
                (k, v) = t
                if v in {'True', 'true'}:
                    return k, True
                elif v in {'False', 'false'}:
                    return k, False

                try:
                    return k, int(v)
                except:
                    try:
                        return k, float(v)
                    except:
                        return k, v

            if ',' in value:
                values = value.split(',')
                return dict(guess_type(tuple(v.split('='))) for v in values)
            else:
                return dict([guess_type(tuple(value.split('=')))])
        else:
            if isinstance(value, dict):
                return value
            else:
                raise TypeError(f"Unsupported type {type(value)}.")


DICT = DictParamType()


# from https://gist.github.com/seansummers/a23276e4c1df990649a6#file-rangeexpand-py
def _group_to_range(group):
    group = "".join(group.split())
    sign, g = ("-", group[1:]) if group.startswith("-") else ("", group)
    r = g.split("-", 1)
    r[0] = "".join((sign, r[0]))
    r = sorted(int(__) for __ in r)
    return range(r[0], 1 + r[-1])


def _range_expand(txt):
    from itertools import chain
    ranges = chain.from_iterable(_group_to_range(__) for __ in txt.split(","))
    return sorted(set(ranges))


class IntRangeListParamType(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            return _range_expand(value)
        else:
            if isinstance(value, list):
                return value
            else:
                return [value]


INT_RANGE_LIST = IntRangeListParamType()


class ListChoice(ListParamType):
    def __init__(self, value):
        assert isinstance(value, (list, Collection))
        self._allowed_values = set(value)

    def convert(self, value, param, ctx):
        converted = super().convert(value, param, ctx)
        values = set(converted)
        if values & self._allowed_values != values:
            raise ValueError(f"Unsupported values: {values & self._allowed_values}")
        return converted


class CombinatorialChoice(ListParamType):
    def __init__(self, value):
        assert isinstance(value, (list, Collection))
        self._allowed_values = set(value)

    def convert(self, value, param, ctx):
        converted = super().convert(value, param, ctx)
        values = set(converted)
        if values & self._allowed_values != values:
            raise ValueError(f"Unsupported values: {values & self._allowed_values}")
        return converted


class DataFrameType(click.Path):

    def __init__(self, *args, **kwargs):
        self.drop_non_numeric = kwargs.pop('drop_non_numeric', False)
        super().__init__(*args, **kwargs)

    def convert(self, value, param, ctx):
        converted = super().convert(value, param, ctx)  # path
        dataframe = pd.read_csv(converted, sep=None, engine="python", index_col=0, header=0,
                                encoding='utf-8')
        non_numeric_columns = [column for column in dataframe.columns
                               if np.dtype(dataframe[column]) in {np.dtype('O')}]
        if self.drop_non_numeric:
            print(f"[INFO] dropping non-numeric columns {', '.join(non_numeric_columns)}.", file=sys.stderr)
            # how exactly does python inheritance work?
            # Why does expr_matrix.drop return an ExpMatrix instead of an ExprMatrix?
            if non_numeric_columns:
                aux = pd.DataFrame(dataframe[non_numeric_columns])
            else:
                aux = None
            dataframe.drop(non_numeric_columns, axis=1, inplace=True)
            return dataframe, aux
        else:
            return dataframe


class Pipe(dict):
    def __init__(self, *arg, **kw):
        super(Pipe, self).__init__(*arg, **kw)
        for key, value in self.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")


pass_pipe = click.make_pass_decorator(Pipe, ensure=True)
Command = namedtuple('Command', 'command params')


def make_cmd_decorator():
    def decorator(f):
        def new_func(*args, **kwargs):
            ctx: click.Context = click.get_current_context()
            pipe = ctx.find_object(Pipe)
            if not hasattr(pipe, 'command'):
                pipe.command = []
            pipe.command.append(Command(ctx.info_name, ctx.params))
            return ctx.invoke(f, pipe, *args[1:], **kwargs)

        return click.decorators.update_wrapper(new_func, f)

    return decorator


arggist = make_cmd_decorator()


def plain_command_string(commands: Iterable[Command]) -> str:
    s = []
    for (cmd, params) in commands:
        if len(s) > 1 and cmd in {'T', 'transpose'} and s[-1] in {'T', 'transpose'}:
            continue
        if params:
            p = [f"{k} = {v}" for k, v in params.items() if v]
            s.append(cmd + ' {' + ', '.join(p) + '}')
        else:
            s.append(cmd)
    return ' → '.join(s)
