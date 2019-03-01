# -*- coding: utf-8 -*-

"""
@author: Guo jingyao <jyguo@smu.edu.sg>

The dataset contains
* 49,290 users who rated a total of
* 139,738 different items at least once, writing
* 664,824 reviews and
* 487,181 issued trust statements.

"""

from ..utils import cache
from ..data import reader

def load_data():
    """Load the ratings given by users to items.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user_id item_id rating_value).
        For example,  23 387 5
        represents the fact "user 23 has rated item 387 as 5"

        Ranges:
            user_id is in [1,49290]
            item_id is in [1,139738]
            rating_value is in [1,5]

    """
    fpath = cache(url='http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2',
                  unzip=False, relative_path='ratings_data/ratings_data.txt.csv')
    return reader.read_uir(fpath, sep=' ', skip_lines=1)


def load_truststate():
    """Load the trust statements issued by users.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (source_user_id target_user_id trust_statement_value).
        For example, the line 22605 18420 1
        represents the fact "user 22605 has expressed a positive trust statement on user 18420"

        Ranges:
           source_user_id and target_user_id are in [1,49290]
           trust_statement_value is always 1 (since in the dataset there are only positive trust statements and not negative ones (distrust)).

    """
    fpath = cache(url='http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2',
                  unzip=False, relative_path='trust_data/trust_data.txt')
    return reader.read_uir(fpath, sep=' ', skip_lines=1)