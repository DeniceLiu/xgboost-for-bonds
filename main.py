# main.py

from model_xgb import load_data, train_xgb
from model_rf import train_rf
from model_nn import train_nn

def evaluate_model(name, mse, r2):
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")


def main():
    df = load_data()

    ### ========== IG MODELING ==========
    features_ig = ['SP500_Volume', 'IG_Lagged', '1x0', '4x0', '12x0']
    target_ig = 'S&P 500 Investment Grade Corporate Bond Index'

    print("=== IG - Random Forest ===")
    mse_rf_ig, r2_rf_ig, y_test_rf_ig, y_pred_rf_ig, model_rf_ig = train_rf(df[features_ig], df[target_ig])
    evaluate_model("Random Forest IG", mse_rf_ig, r2_rf_ig)

    print("=== IG - XGBoost ===")
    mse_xgb_ig, r2_xgb_ig, y_test_xgb_ig, y_pred_xgb_ig, model_xgb_ig = train_xgb(df[features_ig], df[target_ig])
    evaluate_model("XGBoost IG", mse_xgb_ig, r2_xgb_ig)

    # print("=== IG - Neural Network ===")
    # mse_nn_ig, r2_nn_ig, y_test_nn_ig, y_pred_nn_ig, model_nn_ig = train_nn(df[features_ig], df[target_ig])
    # evaluate_model("Neural Net IG", mse_nn_ig, r2_nn_ig)

    ### ========== HY MODELING ==========
    features_hy = ['SP500_Volume', 'HY_Lagged', '1x0', '4x0', '12x0']
    target_hy = 'S&P U.S. Dollar Global High Yield Corporate Bond Index'

    print("\n=== HY - Random Forest ===")
    mse_rf_hy, r2_rf_hy, y_test_rf_hy, y_pred_rf_hy, model_rf_hy = train_rf(df[features_hy], df[target_hy])
    evaluate_model("Random Forest HY", mse_rf_hy, r2_rf_hy)

    print("=== HY - XGBoost ===")
    mse_xgb_hy, r2_xgb_hy, y_test_xgb_hy, y_pred_xgb_hy, model_xgb_hy = train_xgb(df[features_hy], df[target_hy])
    evaluate_model("XGBoost HY", mse_xgb_hy, r2_xgb_hy)

    # print("=== HY - Neural Network ===")
    # mse_nn_hy, r2_nn_hy, y_test_nn_hy, y_pred_nn_hy, model_nn_hy = train_nn(df[features_hy], df[target_hy])
    # evaluate_model("Neural Net HY", mse_nn_hy, r2_nn_hy)


if __name__ == "__main__":
    main()