# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import itertools
from collections import Counter


def read_text(fpath, sep=None, encoding='utf-8', errors=None):
    """Read text file and return two lists of text documents and corresponding ids.
    If `sep` is None, only return one list containing elements are lines of text
    in the original file.

    Parameters
    ----------
    fpath: str
        Path to the data file

    sep: str, default = None
        The delimiter string used to split `id` and `text`. Each line is assumed
        containing an `id` followed by corresponding `text` document.
        If `None`, each line will be a `str` in returned list.

    encoding: str, default = `utf-8`
        Encoding used to decode the file.

    errors: int, default = None
        Optional string that specifies how encoding errors are to be handled.
        Pass 'strict' to raise a ValueError exception if there is an encoding error
        (None has the same effect), or pass 'ignore' to ignore errors.
    """
    with open(fpath, encoding=encoding, errors=errors) as f:
        if sep is None:
            return f.read().splitlines()
        else:
            texts, ids = [], []
            for line in f:
                tokens = line.strip().split(sep)
                ids.append(tokens[0])
                texts.append(sep.join(tokens[1:]))
            return texts, ids


def ui_parser(tokens, line_idx, id_inline=False, **kwargs):
    if id_inline:
        return [(str(line_idx + 1), iid, 1.0) for iid in tokens]
    else:
        return [(tokens[0], iid, 1.0) for iid in tokens[1:]]


def uir_parser(tokens, **kwargs):
    return [(tokens[0], tokens[1], float(tokens[2]))]


PARSERS = {
    'UI': ui_parser,
    'UIR': uir_parser
}


class Reader():
    """Reader class for reading data with different types of format.

    Parameters
    ----------
    user_set: set, default = None
        Set of users to be selected when reading data.
        If `None`, all users that appear in the data will be included.

    item_set: set, default = None
        Set of items to be selected when reading data.
        If `None`, all items that appear in the data will be included.

    min_user_freq: int, default = 1
        The minimum frequency of a user to be selected.
        If `min_user_freq=1`, all users that appear in the data will be included.

    min_item_freq: int, default = 1
        The minimum frequency of an item to be selected.
        If `min_item_freq=1`, all items that appear in the data will be included.

    bin_threshold: float, default = None
        The rating threshold to binarize rating values (turn explicit feedback to implicit feedback).
        For example, if `bin_threshold = 3.0`, all rating values >= 3.0 will be set to 1.0,
        and the rest (< 3.0) will be discarded.

    encoding: str, default = `utf-8`
        Encoding used to decode the file.

    errors: int, default = None
        Optional string that specifies how encoding errors are to be handled.
        Pass 'strict' to raise a ValueError exception if there is an encoding error
        (None has the same effect), or pass 'ignore' to ignore errors.
    """

    def __init__(self, user_set=None, item_set=None, min_user_freq=1, min_item_freq=1,
                 bin_threshold=None, encoding='utf-8', errors=None):
        self.users = user_set
        self.items = item_set
        self.min_uf = min_user_freq
        self.min_if = min_item_freq
        self.bin_threshold = bin_threshold
        self.encoding = encoding
        self.errors = errors

    def filter(self, tuples):
        if self.bin_threshold is not None:
            tuples = [tup for tup in tuples if tup[2] >= self.bin_threshold]

        if self.users is not None:
            if isinstance(self.users, list):
                self.users = set(self.users)
            tuples = [tup for tup in tuples if tup[0] in self.users]

        if self.items is not None:
            if isinstance(self.items, list):
                self.items = set(self.items)
            tuples = [tup for tup in tuples if tup[1] in self.items]

        if self.min_uf > 1:
            user_freq = Counter([tup[0] for tup in tuples])
            tuples = [tup for tup in tuples if user_freq[tup[0]] >= self.min_uf]

        if self.min_if > 1:
            item_freq = Counter([tup[1] for tup in tuples])
            tuples = [tup for tup in tuples if item_freq[tup[1]] >= self.min_if]

        return tuples

    def read(self, fpath, fmt='UIR', sep='\t', skip_lines=0, id_inline=False, parser=None):
        """Read data and parse line by line based on provided `fmt` or `parser`.

        Parameters
        ----------
        fpath: str
            Path to the data file

        fmt: str, default: `UIR`
            Line format to be parsed

        sep: str, default: \t
            The delimiter string.

        skip_lines: int, default: 0
            Number of first lines to skip

        id_inline: bool, default: False
            If `True`, user ids corresponding to the line numbers of the file,
            where all the ids in each line are item ids.

        parser: function, default: None
            Function takes a list of `str` tokenized by `sep` and
            returns a list of tuples which will be joined to the final results.
            If `None`, parser will be determined based on `fmt`.

        Returns
        -------
        tuples: list
            Data in the form of list of tuples. What inside each tuple
            depends on `parser` or `fmt`.

        """
        parser = PARSERS.get(fmt.upper(), None) if parser is None else parser
        if parser is None:
            raise ValueError('Invalid line format: {}\n'
                             'Only support: {}'.format(fmt, PARSERS.keys()))
        with open(fpath, encoding=self.encoding, errors=self.errors) as f:
            tuples = [tup
                      for idx, line in enumerate(itertools.islice(f, skip_lines, None))
                      for tup in parser(line.strip().split(sep), line_idx=idx, id_inline=id_inline)]
            return self.filter(tuples)
