from __future__ import print_function

import time

import logging
import numpy
import pandas
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, NMSLibAlternatingLeastSquares,
                                      FaissAlternatingLeastSquares)
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)

logging.basicConfig(level=logging.DEBUG)

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bm25": BM25Recommender}


def get_model(model_name):
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 50, 'dtype': numpy.float32}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    else:
        params = {}

    return model_class(**params)


if __name__ == '__main__':

    start = time.time()
    data = pandas.read_csv("/Users/zp/Downloads/7day_data",
                             usecols=[1, 3],
                             names=['artist', 'user'],
                             na_filter=False)

    # map each artist and user to a unique numeric value
    data['plays'] = 1
    data['user'] = data['user'].astype("category")
    data['artist'] = data['artist'].astype("category")

    logging.info("start1")

    # create a sparse matrix of all the users/plays
    item_user_data = coo_matrix((data['plays'].astype(float),
                       (data['artist'].cat.codes.copy(),
                        data['user'].cat.codes.copy())))

    # initialize a model
    logging.info("initialize a model")
    model = get_model("als")

    logging.info("initialized model")

    # train the model on a sparse matrix of item/user/confidence weights
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        item_user_data = bm25_weight(item_user_data, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    model.fit(item_user_data)

    # # recommend items for a user
    #
    artists = dict(enumerate(data['artist'].cat.categories))
    # # user = dict(enumerate(data['user'].cat.categories))

    user_items = item_user_data.T.tocsr()

    # logging.info(data['user'])

    uid = 8000025933

    dict_user = {value:key for key,value in enumerate(data['user'].cat.categories)}

    if dict_user.has_key(uid):
         for artistid, score in model.recommend(dict_user[uid], user_items, 200):
             logging.info("%s\t%s\t%s\n" % (1, artists[artistid], score))
    else:
        logging.error("uid not in recommendlist")
