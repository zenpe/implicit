from __future__ import print_function

import os

import logging
import numpy
import pandas
import json

from urlparse import urlparse, parse_qs
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, NMSLibAlternatingLeastSquares, FaissAlternatingLeastSquares)
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,TFIDFRecommender, bm25_weight)

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from apscheduler.schedulers.background import BackgroundScheduler


logging.basicConfig(level=logging.DEBUG)

file_path = "/Users/zp/Downloads/7day_data"
if os.environ.get("DAY7_DATA"):
    file_path = os.environ.get("DAY7_DATA")

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bm25": BM25Recommender}

class ReccomendHandler(BaseHTTPRequestHandler):
    data = None
    user_items = None

    def do_HEAD(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def validate_param(self):

        if self.path.startswith('/health'):
            self.send_response(200, "ok")
            return None, None
        elif not self.path.startswith('/recommend'):
            self.send_response(404, "Invalid request")
            return None, None

        query_components = parse_qs(urlparse(self.path).query)
        uid = query_components.get("uid", None)
        count = query_components.get("count", None)

        if uid is None:
            self.send_error(404, "uid must be in params")
            return None, None
        else:
            uid = uid[0]

        if count is None:
            count = 20
        else:
            count = int(count[0])

        return uid, count

    def do_GET(self):

        uid, count = self.validate_param()
        if uid is None or count is None:
            return
        articles = dict(enumerate(self.data['article'].cat.categories))
        dict_user = {value:key for key,value in enumerate(self.data['user'].cat.categories)}

        items = []
        if dict_user.has_key(uid):
             for itemid, score in self.model.recommend(dict_user[uid], self.user_items, count):
                 items.append(articles[itemid])
                 # logging.info("%s\t%s\t%s\n" % (1, artists[artistid], score))
        else:
            logging.error("uid not in recommendlist")
            self.send_error(404, "uid not in recommendlist")

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"recommendations":items}))

def read_data(file_path):
    data = pandas.read_csv(file_path,
                usecols=[1, 3],
                names=['article', 'user'],
                na_filter=False,
                error_bad_lines=False)

    # map each article and user to a unique numeric value
    data['views'] = 1
    data['user'] = data['user'].astype("category")
    data['article'] = data['article'].astype("category")
    return data

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


def load_model(data):

    logging.info("load model begin...")
    # create a sparse matrix of all the users/views
    item_user_data = coo_matrix((data['views'].astype(float), (data['article'].cat.codes.copy(), data['user'].cat.codes.copy())))
    logging.info("initialize a model")

    model = get_model("als")
    # train the model on a sparse matrix of item/user/confidence weights
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        item_user_data = bm25_weight(item_user_data, K1=100, B=0.8)
        # also disable building approximate recommend index
        model.approximate_similar_items = False
    model.fit(item_user_data)
    user_items = item_user_data.T.tocsr()
    logging.info("load model finished...")

    return model, user_items

def load_model_timely():
    ReccomendHandler.data = read_data(file_path)
    ReccomendHandler.model , ReccomendHandler.user_items = load_model(ReccomendHandler.data)


if __name__ == '__main__':

    load_model_timely()

    # BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(load_model_timely, 'interval', seconds = 600)
    scheduler.start()

    server = HTTPServer(('localhost', 8888), ReccomendHandler)
    logging.info("Starting server, use <Ctrl-C> to stop")
    server.serve_forever()



