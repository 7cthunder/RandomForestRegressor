import math
import multiprocessing
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython

cdef struct tree_node:
    double value
    double split_value
    int split_index
    int f_idx
    tree_node *left
    tree_node *right

cdef tree_node *TreeNode():
    cdef tree_node *t
    t = <tree_node *>PyMem_Malloc(sizeof(tree_node))
    t.f_idx = -1
    t.left  = NULL
    t.right = NULL
    return t

cdef void free_decision_tree_node(tree_node *self):
    PyMem_Free(self)

cdef inline void decision_tree_dealloc(tree_node *self):
    if self != NULL:
        decision_tree_dealloc(self.left)
        decision_tree_dealloc(self.right)
        free_decision_tree_node(self)

cdef double calculate_std(n, s1, s2):
    cdef double var = (s2/n) - (s1/n)**2
    # for computational accuracy loss
    if var < 0: var = 0
    return math.sqrt(var)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double find_better_split(
    tree_node *self, int f_idx, int[:] sort_idx, double[:] X, double[:] y, double best_score, int min_samples_leaf):
    """
    @Params:
        f_idx: integer, feature index
        sort_idx: np.array of shape (#samples, ), index sorted by feature[f_idx]
        X: np.array of shape (#samples, ), value of feature[f_idx]
        y: np.array of shape (#samples, ), value of samples
        best_score: the best split score of tree_node 'self' currently
    @Returns:
        best_score: the best split score of tree_node 'self' currently
    """
    # for cache friendly
    cdef int i
    cdef double[:] sort_X, sort_y
    sort_X = np.empty_like(X)
    sort_y = np.empty_like(y)
    for i in range(sort_idx.shape[0]):
        sort_X[i] = X[sort_idx[i]]
        sort_y[i] = y[sort_idx[i]]
        # print(sort_X[i])

    cdef int lnode_n, rnode_n
    cdef double lnode_sum, lnode_sum2, rnode_sum, rnode_sum2
    lnode_n    = 0
    lnode_sum  = 0.0
    lnode_sum2 = 0.0
    rnode_n    = sort_y.shape[0]
    rnode_sum  = sum(sort_y)
    rnode_sum2 = sum(np.power(sort_y, 2))

    cdef double xi, yi
    cdef double lnode_std, rnode_std, current_score

    for i in range(sort_y.shape[0] - min_samples_leaf):
        xi = sort_X[i]
        yi = sort_y[i]

        lnode_n += 1
        rnode_n -= 1

        lnode_sum += yi
        rnode_sum -= yi

        lnode_sum2 += yi**2
        rnode_sum2 -= yi**2

        # make sure each node has at least 'min_samples_leaf' samples
        if i < (min_samples_leaf - 1) or xi == sort_X[i+1]:
            continue

        lnode_std = calculate_std(lnode_n, lnode_sum, lnode_sum2)
        rnode_std = calculate_std(rnode_n, rnode_sum, rnode_sum2)
        current_score = lnode_std*lnode_n + rnode_std*rnode_n
        if current_score < best_score:
            self.f_idx = f_idx
            self.split_value = xi
            self.split_index = i
            best_score = current_score
            # print(best_score)

    return best_score

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void build_decision_tree(
    tree_node *self, double[:, :] X , double[:] y, 
    int n_features, int max_depth, int min_samples_leaf, int depth = 0):
    """
    Recursively build the decision tree
    """
    # print(f'depth:{depth} max_depth:{max_depth} y.shape[0]:{y.shape[0]} min_samples_leaf * 2:{min_samples_leaf * 2}')
    if depth >= max_depth or y.shape[0] < min_samples_leaf * 2:
        self.value = np.mean(y)
        return
    
    # print('here')

    cdef int i

    # sampling n features of X
    cdef int[:] f_idxs
    cdef int[:, :] sort_idxs
    cdef double[:, :] sampled_X = np.empty((X.shape[0], n_features), dtype=np.double)
    f_idxs = np.random.permutation(X.shape[1])[:n_features].astype(np.int)
    for i in range(n_features):
        sampled_X[:, i] = X[:, f_idxs[i]]
    sort_idxs = np.argsort(sampled_X, axis=0).astype(np.int) # maybe here can be optimized

    # the best split score of tree_node 'self' currently
    # initially large enough
    cdef double best_score = 1e9
    
    for i in range(n_features):
        best_score = find_better_split(self, f_idxs[i], sort_idxs[:, i], sampled_X[:, i], y, best_score, min_samples_leaf)

    # check if has any possible split point
    # if not, just make it a leaf node
    if self.f_idx == -1:
        self.value = np.mean(y)
        return

    cdef int idx
    cdef double[:, :] sort_X
    cdef double[:]    sort_y
    # find the index of f_idx in selected features
    for i in range(n_features):
        if self.f_idx == f_idxs[i]:
            idx = i
            break
    sort_X = np.empty_like(X)
    sort_y = np.empty_like(y)
    for i in range(sort_idxs.shape[0]):
        sort_X[i] = X[sort_idxs[i, idx]]
        sort_y[i] = y[sort_idxs[i, idx]]

    cdef double[:, :] lnode_X, rnode_X
    cdef double[:]    lnode_y, rnode_y
    lnode_X = sort_X[:self.split_index+1]
    rnode_X = sort_X[self.split_index+1:]
    lnode_y = sort_y[:self.split_index+1]
    rnode_y = sort_y[self.split_index+1:]

    # create children node
    self.left = TreeNode()
    build_decision_tree(self.left, lnode_X, lnode_y, n_features, max_depth, min_samples_leaf, depth+1)
    self.right = TreeNode()
    build_decision_tree(self.right, rnode_X, rnode_y, n_features, max_depth, min_samples_leaf, depth+1)

cdef class DecisionTree:
    """
    A simple decision tree
    """
    cdef int n_features
    cdef int max_depth
    cdef int min_samples_leaf
    cdef tree_node *root

    def __cinit__(DecisionTree self, int n_features, int max_depth, int min_samples_leaf):
        self.root = NULL
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def __dealloc__(DecisionTree self):
        decision_tree_dealloc(self.root)

    def fit(DecisionTree self, double[:, :] X, double[:] y):
        # if we had built a tree, we should free the memory of the old tree first
        if self.root is not NULL:
            decision_tree_dealloc(self.root)
        
        self.root = TreeNode()
        build_decision_tree(self.root, X, y, self.n_features, self.max_depth, self.min_samples_leaf)
        
    def predict(DecisionTree self, double[:, :] X):
        if self.root is NULL:
            print('Please call the fit function first.')
            return
        
        cdef int i
        cdef double[:] x
        cdef tree_node *current_node
        preds = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            x = X[i]
            current_node = self.root
            # print(current_node.value)
            # print(current_node.f_idx)
            # just check one child is ok
            while current_node.left is not NULL:
                
                if x[current_node.f_idx] <= current_node.split_value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            preds[i] = current_node.value
        
        return preds