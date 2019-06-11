from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as skrfg
import time
from RandomForestRegressor import RandomForestRegressor
from sklearn.metrics import r2_score

if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rf = RandomForestRegressor()
    start = time.time()
    rf.fit(X_train, y_train)
    end = time.time()
    print(f'my_rf cost: {end - start}s')

    preds = rf.predict(X_test)

    print(f'my_rf score: {r2_score(y_test, preds)}')

