# model_rf.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_rf(X, y, param_grid=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid or {
        'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2],
        'min_samples_leaf': [1], 'bootstrap': [True]
    }, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), y_test, y_pred, grid.best_estimator_

