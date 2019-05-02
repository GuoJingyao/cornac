# -*- coding: utf-8 -*-

"""
Example to run Probabilistic Matrix Factorization (PMF) model with Ratio Split evaluation strategy

@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import cornac
import numpy as np
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import PMF_seperable
from cornac.models import PMF

# Load the MovieLens 100K dataset
ml_100k = movielens.load_100k()

# Instantiate an evaluation method.
ratio_split = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False)

# Instantiate a PMF recommender model.
pmf = PMF_seperable(k=10, max_iter=10, learning_rate=0.001, lamda=0.001,verbose=True,
                    init_params={'V':np.random.normal(loc=0.0, scale=1.0, size=(2000, 10))}, fixedParameter="V")
# pmf = PMF(k=10, max_iter=100, learning_rate=0.001, lamda=0.001)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)
auc = cornac.metrics.AUC()

# Instantiate and then run an experiment.
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[pmf],
                        metrics=[mae, rmse, rec_20, pre_20, auc],
                        user_based=True)
exp.run()