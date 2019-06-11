from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as skrfg
import time
import matplotlib.pyplot as plt
from RandomForestRegressor import RandomForestRegressor
from sklearn.metrics import r2_score

if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rf1 = RandomForestRegressor()
    start = time.time()
    rf1.fit(X_train, y_train)
    end = time.time()
    print(f'my_rf_1 cost: {end - start}s')

    # rf2 = RandomForestRegressor(n_jobs=2)
    # start = time.time()
    # rf2.fit(X_train, y_train)
    # end = time.time()
    # print(f'my_rf_2 cost: {end - start}s')

    # rf3 = RandomForestRegressor(n_jobs=3)
    # start = time.time()
    # rf3.fit(X_train, y_train)
    # end = time.time()
    # print(f'my_rf_3 cost: {end - start}s')

    # rf4 = RandomForestRegressor(n_jobs=4)
    # start = time.time()
    # rf4.fit(X_train, y_train)
    # end = time.time()
    # print(f'my_rf_4 cost: {end - start}s')

    # skrf = skrfg(n_estimators=100, n_jobs=4)
    # start = time.time()
    # skrf.fit(X_train, y_train)
    # end = time.time()
    # print(f'sk_rf_4 cost: {end - start}s')

    preds1 = rf1.predict(X_test)
    # preds2 = rf2.predict(X_test)
    # preds3 = rf3.predict(X_test)
    # preds4 = rf4.predict(X_test)
    # preds5 = skrf.predict(X_test)

    # plt.plot(y_test, label='ground_truth')
    # plt.plot(preds, label='my_rf')
    # plt.plot(preds2, label='sk_rf')

    # plt.legend()
    # plt.show()

    print(f'my_rf_1 score: {r2_score(y_test, preds1)}')
    # print(f'my_rf_2 score: {r2_score(y_test, preds2)}')
    # print(f'my_rf_3 score: {r2_score(y_test, preds3)}')
    # print(f'my_rf_4 score: {r2_score(y_test, preds4)}')
    # print(f'sk_rf_4 score: {r2_score(y_test, preds5)}')
