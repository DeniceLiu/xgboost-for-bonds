# model_xgb.py

import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def load_data(start="2021-02-01", end="2024-02-29"):
    sp500_data = yf.download("^GSPC", start=start, end=end, auto_adjust=False)['Adj Close']
    sp500_data.rename(columns={'^GSPC': 'SP500_Adj_Close'}, inplace=True)

    ig_data = pd.read_csv('IG.csv', index_col='Date', parse_dates=True)
    hy_data = pd.read_csv('HY.csv', index_col='Date', parse_dates=True)

    combined = sp500_data.join(ig_data, how='left').join(hy_data, how='left', rsuffix='_HY')

    combined['SP500_Daily_Return'] = combined['SP500_Adj_Close'].pct_change()
    combined['IG_Daily_Return'] = combined['S&P 500 Investment Grade Corporate Bond Index'].pct_change()
    combined['HY_Daily_Return'] = combined['S&P U.S. Dollar Global High Yield Corporate Bond Index'].pct_change()

    combined.fillna(method='ffill', inplace=True)
    combined.fillna(method='bfill', inplace=True)

    sp500_vol = yf.download("^GSPC", start=start, end=end)['Volume']
    combined['SP500_Volume'] = sp500_vol

    combined['Recent_5D_Avg'] = combined['SP500_Adj_Close'].rolling(window=5).mean()
    combined['1x0'] = combined['Recent_5D_Avg'] / combined['Recent_5D_Avg'].shift(5)
    combined['4x0'] = combined['Recent_5D_Avg'] / combined['Recent_5D_Avg'].shift(20)
    combined['12x0'] = combined['Recent_5D_Avg'] / combined['Recent_5D_Avg'].shift(60)

    combined['IG_Lagged'] = combined['S&P 500 Investment Grade Corporate Bond Index'].shift(5)
    combined['HY_Lagged'] = combined['S&P U.S. Dollar Global High Yield Corporate Bond Index'].shift(5)

    combined.dropna(inplace=True)
    return combined

def train_xgb(X, y, param_grid=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid = GridSearchCV(model, param_grid or {
        'n_estimators': [100], 'max_depth': [6], 'learning_rate': [0.1],
        'subsample': [0.8], 'colsample_bytree': [0.8]
    }, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), y_test, y_pred, grid.best_estimator_
