from typing import Optional
from tqdm import tqdm
from types import MethodType

try:
    _range = xrange
except NameError:
    _range = range


def set_description(self, _: str):
    # if rank != 0, output nothing!
    pass


def etqdm(iterable, rank: Optional[int] = None, **kwargs):
    if rank:
        iterable.set_description = MethodType(set_description, iterable)
        return iterable
    else:
        return tqdm(iterable, bar_format="{l_bar}{bar:3}{r_bar}", colour="#ffa500", **kwargs)


def etrange(*args, **kwargs):
    """
    A shortcut for tqdm(xrange(*args), **kwargs).
    On Python3+ range is used instead of xrange.
    """
    return etqdm(_range(*args), **kwargs)
