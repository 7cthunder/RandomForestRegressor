import multiprocessing
import numpy as np
from DecisionTree import DecisionTree

class RandomForestRegressor(object):
    """
    An toy implementation of RandomForestRegressor in sklearn
    """
    def __init__(self, n_estimators=100, n_features='auto', n_samples_ratio=0.6, max_depth=None, min_samples_leaf=1, n_jobs=1):
        """
        @Forest Params:
            n_estimators: number of trees
            n_samples_ratio: ratio of samples for tree building
        @Tree Params:
            n_features: number of features sampled for tree building (default='auto' represents using all features)
            max_depth: maximum depth for every tree (default=None represents no limitation)
            min_samples_leaf: minimum number of samples for each node of tree
        @Other Params:
            n_jobs: number of process for fitting and predicting
        """
        self.n_estimators     = n_estimators
        self.n_features       = n_features
        self.n_samples_ratio  = n_samples_ratio
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf

        if self.max_depth == None:
            self.max_depth = 0xfff 

        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        @Params:
            X: a numpy array of size (#samples, #features) 
            y: a numpy array of size (#samples, ) 
        """
        if self.n_features == 'auto':
            self.n_features = X.shape[1]

        # mgr = multiprocessing.Manager()
        # self.trees = mgr.list()
        self.trees = []

        pool = multiprocessing.Pool(self.n_jobs)

        # currently implement a sequential version
        for i in range(self.n_estimators):
            # print(i)
            sample_idxs = np.random.permutation(y.shape[0])[:round(self.n_samples_ratio * y.shape[0])]
            # pool.apply_async(self.build_tree, args=(X[sample_idxs, :], y[sample_idxs]))
            self.build_tree(X[sample_idxs, :], y[sample_idxs])

        # pool.close()
        # pool.join()

    def build_tree(self, X, y):
        tree = DecisionTree(
            self.n_features,
            self.max_depth,
            self.min_samples_leaf
        )
        # print(1)
        tree.fit(X, y)
        # print(tree)
        self.trees.append(tree)

    def predict(self, X):
        """
        @Params:
            X: a numpy array of size (#samples, #features) for predicting
        @Returns:
            pred: a numpy array of size (#samples, ) for result
        """
        return np.mean([t.predict(X) for t in self.trees], axis=0)