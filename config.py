from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

DATASETS_DIRECTORY = "datasets"
DATASET_FILENAME = "data.csv"
OUTPUT_DIRECTORY = "output"

TARGET_FEATURE = "USDBRL Curncy"
TARGET = "Target"
FEATURES = [
    "Date",
    "EURUSD Curncy",
    "GBPUSD Curncy",
    "USDJPY Curncy",
    "USDMXN Curncy",
    "USOSFR2 Curncy",
    "USOSFR10 Curncy",
    "CO1 Comdty",
    "CU1 Comdty",
    "XAU Curncy",
    "BCNI3M Curncy",
    "VIX Index",
    "ES1 Index",
    "NQ1 Index",
    "IBOV Index",
    "DXY Curncy",
    "BRAZIL CDS USD SR 5Y D14 Corp",
    "MEX CDS USD SR 5Y D14 Corp",
    "EURUSDV1M BGN Curncy",
    "W 1 COMB Comdty",
    "C 1 COMB Comdty",
    "KC1 Comdty",
    "USGGBE2 Index",
    "USGGBE10 Index",
    "CESIUSD Index",
]

ROLLING_WINDOW = 30

TARGET_RANGE_MIN = -5
TARGET_RANGE_MAX = 5

MODELS = [
    {LinearRegression: {"fit_intercept": [True, False]}},
    {
        KNeighborsRegressor: {
            "n_neighbors": [1, 3, 5, 8],
            "leaf_size": [5, 10, 30, 50],
        }
    },
    {
        SVR: {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        }
    },
    {DecisionTreeRegressor: {"max_depth": [3, 5, 10]}},
    {
        RandomForestRegressor: {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
        }
    },
    {
        GradientBoostingRegressor: {
            "n_estimators": [100, 200, 300],
            "learning_rate": [1, 0.1, 0.01],
            "max_depth": [3, 5, 10],
        }
    },
    {
        AdaBoostRegressor: {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.01, 0.001],
        }
    },
    {
        XGBRegressor: {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.1, 0.01, 0.001],
            "max_depth": [3, 5, 10],
        }
    },
    {
        LGBMRegressor: {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.1, 0.01, 0.001],
            "max_depth": [3, 5, 10],
        }
    },
    {
        CatBoostRegressor: {
            "learning_rate": [0.1, 0.01, 0.001],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10],
            "subsample": [1.0],
            "colsample_bylevel": [1.0],
            "reg_lambda": [3.0],
        }
    },
]
