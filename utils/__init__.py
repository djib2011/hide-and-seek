from contextlib import contextmanager
import sys
import os

from . import datagen


@contextmanager
def suppress_stdout():
    """
    Function for 'with' statement context managers. It is meant to suppress the output of anything inside the statement.
    e.g:
    >>> print('1')
    >>> with suppress_stdout():
    >>>    print('2')
    >>> print('3')
    1
    3
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
