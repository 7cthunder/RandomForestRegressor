# RandomForestRegressor
An toy implementation of RandomForestRegressor in sklearn

## Python Version
The python version support multi-processing, you can explicitly give a int value for the `n_jobs`, which means how many process the RandomForest will use.

### Usage
You can use it in this way:
```python
from RandomForestRegressor import RandomForestRegressor
# new an instance
rf = RandomForestRegressor(n_estimators=100)
# fitting for the training data
rf.fit(X, y)
# predict
preds = rf.predict(X)
```

## Cython Version
A cython version of the random forest had been implemented, however, it can't use the multiprocessing module to build trees concurrently. If you know what happen, please open an issue.

### Usage
For cython program need to be compiled first, please use the following command to do that:
```bash
python setup.py build_ext --inplace
```

Then you can import and use the RandomForestRegressor like the python version:
```python
from RandomForestRegressor import RandomForestRegressor
# new an instance
rf = RandomForestRegressor(n_estimators=100)
# fitting for the training data
rf.fit(X, y)
# predict
preds = rf.predict(X)
```