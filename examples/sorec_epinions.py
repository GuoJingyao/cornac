import cornac
from cornac.datasets import epinions
from cornac.eval_methods import RatioSplit
from cornac.data.graph import *

# Load the MovieLens 100K dataset
# data = epinions.load_data()

uir_triplets = []
with open("ratings_data.txt", 'r') as f:
    for line in f:
        tokens = [token.strip() for token in line.split(' ')]
        uir_triplets.append((tokens[0], tokens[1], float(tokens[2])))

rating = uir_triplets

uir_triplets = []
with open("trust_data.txt", 'r') as f:
    for line in f:
        tokens = [token.strip() for token in line.split(' ')]
        uir_triplets.append((tokens[1], tokens[2], float(tokens[3])))
trust = uir_triplets

# data = TrainSet.uir_tuple(rating)
user_graph_module = GraphModule(data = trust, normalized=True)

ratio_split = RatioSplit(data=rating,
                         test_size=0.2, rating_threshold=0.5,
                         exclude_unknowns=True, verbose=True,
                         user_graph=user_graph_module)

sorec = cornac.models.SOREC(l=5, max_iter=100, learning_rate=0.001, lamda_C=10, lamda=0.001, init_params={'U': None, 'V': None, 'Z': None})

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
exp = cornac.Experiment(eval_method=ratio_split,
                        models=[sorec],
                        metrics=[mae, rmse, rec_20, pre_20],
                        user_based=True)
exp.run()