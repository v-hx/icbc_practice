import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

from config import ROLLING_WINDOW, TARGET_RANGE_MAX


class Regressor:
    def __init__(self, data, model, params, X_train, y_train):
        self.data = data
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train

    def create_grid_search(self, scoring="neg_mean_absolute_error"):
        return GridSearchCV(
            estimator=self.model, param_grid=self.params, scoring=scoring
        )

    def fit(self, grid_search, X_train_window, y_train_window):
        grid_search.fit(X_train_window, y_train_window)

    def get_best_model(self, grid_search):
        return grid_search.best_estimator_

    def get_best_params(self, grid_search):
        return grid_search.best_params_

    def get_mae(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def predict(self, grid_search, X_predict):
        best_model = self.get_best_model(grid_search)
        return best_model.predict(X_predict).round()[0]

    def train(self, collector):
        range_limit = self.X_train.shape[0] - ROLLING_WINDOW

        grid_search = self.create_grid_search()

        for i in range(0, range_limit):
            start_time = time.time()
            chunk = i + ROLLING_WINDOW

            collector.uuids.append(i + 1)
            y_true = self.y_train.iloc[chunk]
            collector.dates.append(self.data.iloc[chunk]["Date"])
            collector.y_trues.append(y_true)

            X_train_window = self.X_train[i:chunk]
            y_train_window = self.y_train[i:chunk]

            # Train and predict
            self.fit(grid_search, X_train_window, y_train_window)
            y_pred = self.predict(grid_search, self.X_train[chunk : chunk + 1])
            collector.y_preds.append(y_pred)

            # Calculate stats
            collector.maes.append(self.get_mae([y_true], [y_pred]))
            collector.best_params.append(self.get_best_params(grid_search))

            # Calculate estimators
            change = self.data.iloc[chunk]["Change"]
            collector.changes.append(change)

            pnl = y_pred / TARGET_RANGE_MAX * change
            collector.pnls.append(pnl)

            end_time = time.time()
            collector.seconds_taken.append(round((end_time - start_time), 3))
