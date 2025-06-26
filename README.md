# Bond Index Forecasting using Equity Momentum Factor with XGBoost, Random Forest, and Neural Networks

This repository implements a machine learning pipeline to forecast corporate bond indices — specifically the **S&P 500 Investment Grade (IG)** and **High Yield (HY)** bond indices — using macro-financial features derived from the S&P 500 and historical bond prices.

### Data

- **Equity Index**: S&P 500 (adjusted close, volume)
- **Bond Indices**: IG and HY indices from CSV files
- **Features Engineered**:
  - Lagged bond index values
  - SP500 return ratios: 1x0, 4x0, 12x0 (momentum-like signals)
  - Rolling 5-day averages

### Models

- `XGBoost`: Tuned via `GridSearchCV` using R² as scoring metric.
- `Random Forest`: Used to compare tree-based ensemble performance.
- `Neural Network`: (Optional) Tuned with Keras Tuner for experimentation.

### Evaluation Metrics

- **MSE (Mean Squared Error)**
- **R² (Coefficient of Determination)**

### Key Findings

XGBoost consistently outperformed Random Forest in predicting both IG and HY index values, achieving higher R² and lower MSE.

For the IG index, XGBoost reached an R² of 0.8256, significantly better than Random Forest's 0.7256.

For the HY index, both models performed well, with XGBoost achieving R² of 0.9151 and Random Forest close behind at 0.9172.

Momentum features like 1x0, 4x0, and lagged bond values contributed significantly to model performance.

Feature importance analysis indicated that lagged bond values and rolling momentum ratios were strong predictors for both IG and HY.

### References
- Hendrik Kaufmann, Philip Messow & Jonas Vogt (2021) Boosting the Equity
Momentum Factor in Credit, Financial Analysts Journal, 77:4, 83-103, DOI:
10.1080/0015198X.2021.1954377
- Dor, Arik Ben, and Zhe Xu. 2015. “Should Equity Investors Care about
Corporate Bond Prices? Using Bond Prices to Construct Equity Momentum
Strategies.” Journal of Portfolio Management 41 (4): 35–49.
- Gebhardt, William R., Soeren Hvidkjaer, and Bhaskaran Swaminathan. 2005.
“Stock and Bond Market Interaction: Does Momentum Spill Over?” Journal of
Financial Economics 75 (3): 651–90.
