# Retail Demand Forecasting

An end-to-end Data Science project using the Kaggle Store Sales competition.

## Objective
Forecast store and product-family sales using historical demand, calendar features,
promotions, store attributes, oil prices, and holiday information.

## Version 1 results

| Metric | Result |
|---|---:|
| Local validation RMSLE | 0.42988 |
| Seasonal baseline RMSLE | 0.61704 |
| Improvement over baseline | 30.33% |
| Official Kaggle score | 0.51413 |

The local validation and official Kaggle scores come from different evaluation sets.

## Methods
- Data quality auditing and exploratory data analysis
- Calendar and store features
- Lagged sales and rolling averages
- LightGBM regression
- Chronological, recursive 16-day validation
- RMSLE evaluation and feature importance

## Project structure
```text
retail-demand-forecasting/
├── data/raw/
├── data/processed/
├── notebooks/
├── models/
├── reports/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset
Download the data from the [Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
Raw competition files are intentionally excluded from GitHub.

## Run the dashboard
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Future improvements
- Improve store-specific holiday matching
- Compare additional forecasting models
- Tune model hyperparameters
- Evaluate errors across stores and forecast horizons

## Acknowledgements
Dataset: Corporación Favorita Store Sales competition on Kaggle.
