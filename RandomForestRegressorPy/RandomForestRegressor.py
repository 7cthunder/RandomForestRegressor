import numpy as np
import math
import multiprocessing

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
        mgr = multiprocessing.Manager()
        self.trees = mgr.list()

        pool = multiprocessing.Pool(self.n_jobs)

        # currently implement a sequential version
        for i in range(self.n_estimators):
            sample_idxs = np.random.permutation(y.shape[0])[:round(self.n_samples_ratio * y.shape[0])]
            pool.apply_async(self.build_tree, args=(X[sample_idxs, :], y[sample_idxs]))

        pool.close()
        pool.join()

    def build_tree(self, X, y):
        tree = DecisionTree(
            self.n_features,
            self.max_depth,
            self.min_samples_leaf
        )
        tree.fit(X, y)
        self.trees.append(tree)

    def predict(self, X):
        """
        @Params:
            X: a numpy array of size (#samples, #features) for predicting
        @Returns:
            pred: a numpy array of size (#samples, ) for result
        """
        return np.mean([t.predict(X) for t in self.trees], axis=0)

class DecisionTree(object):
    def __init__(self, n_features='auto', max_depth=None, min_samples_leaf=1):
        """
        @Params:
            n_features: number of features sampled for tree building (default='auto', represents using all features)
            max_depth: maximum depth for every tree (default=None, represents no limitation)
            min_samples_leaf: minimum number of samples for each node of tree
        """
        self.n_features       = n_features
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.root = None

    def fit(self, X, y):
        self.root = TreeNode(X, y, self.n_features, self.max_depth, self.min_samples_leaf)
    
    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            current_node = self.root
            while not current_node.is_leaf:
                if x[current_node.f_idx] <= current_node.split['xi']:
                    current_node = current_node.lnode
                else:
                    current_node = current_node.rnode
            preds[i] = current_node.val
        return preds


class TreeNode(object):
    def __init__(self, X, y, n_features, max_depth, min_samples_leaf, depth=0):
        self.depth   = depth         # depth of the tree node
        self.score   = float('inf')  # score of split 
        self.f_idx   = -1            # the index of split feature
        self.split   = None          # the split point, a dict object { 'xi': ..., 'idx': ... }
        self.is_leaf = False         # the flag for checking if the tree node is a leaf or not
        self.val     = None          # the value of the tree node
        
        self.n_features       = n_features
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf

        if depth >= max_depth or y.shape[0] < min_samples_leaf * 2:
            self.is_leaf = True
            self.val = np.mean(y)
            return

        if n_features == 'auto':
            n_features = X.shape[1]
        
        # sample n features
        f_idxs    = np.random.permutation(X.shape[1])[:n_features]
        sort_idxs = np.argsort(X[:, f_idxs], axis=0) # maybe here can be optimized
        
        for i, f_idx in enumerate(f_idxs):
            self.find_better_split(f_idx, sort_idxs[:, i], X[:, f_idx], y)

        # check if has any possible split point
        if self.f_idx == -1:
            self.is_leaf = True
            self.val = np.mean(y)
            return

        # the index of f_idx in f_idxs
        idx = np.where(f_idxs==self.f_idx)[0][0]
        sort_X, sort_y = X[sort_idxs[:, idx]], y[sort_idxs[:, idx]]

        lnode_X, rnode_X = sort_X[:self.split['idx']+1], sort_X[self.split['idx']+1:]
        lnode_y, rnode_y = sort_y[:self.split['idx']+1], sort_y[self.split['idx']+1:]

        # create children node 
        self.lnode = TreeNode(lnode_X, lnode_y, self.n_features, self.max_depth, self.min_samples_leaf, depth=self.depth+1)
        self.rnode = TreeNode(rnode_X, rnode_y, self.n_features, self.max_depth, self.min_samples_leaf, depth=self.depth+1)

    def _std(self, n, s1, s2):
        var = (s2/n) - (s1/n)**2
        # for computational accuracy loss
        if var < 0: var = 0
        return math.sqrt(var)

    def find_better_split(self, f_idx, sort_idx, X, y):
        """
        @Params:
            f_idx: integer, feature index
            sort_idx: np.array of shape (#samples, ), index sorted by feature[f_idx]
            X: np.array of shape (#samples, ), value of feature[f_idx]
            y: np.array of shape (#samples, ), value of samples
        """
        # for cache friendly
        sort_y = y[sort_idx]
        sort_X = X[sort_idx]

        lnode_n, lnode_sum, lnode_sum2 = 0, 0.0, 0.0
        rnode_n, rnode_sum, rnode_sum2 = len(y), sort_y.sum(), (sort_y**2).sum() 

        for i in range(len(y) - self.min_samples_leaf):
            xi, yi = sort_X[i], sort_y[i]

            lnode_n += 1
            rnode_n -= 1

            lnode_sum += yi
            rnode_sum -= yi

            lnode_sum2 += yi**2
            rnode_sum2 -= yi**2

            # make sure each node has at least 'min_samples_leaf' samples
            if i < (self.min_samples_leaf - 1) or xi == sort_X[i+1]:
                continue

            lnode_std = self._std(lnode_n, lnode_sum, lnode_sum2)
            rnode_std = self._std(rnode_n, rnode_sum, rnode_sum2)
            current_score = lnode_std*lnode_n + rnode_std*rnode_n
            if current_score < self.score:
                self.f_idx = f_idx
                self.score = current_score
                self.split = {'xi': xi, 'idx': i}