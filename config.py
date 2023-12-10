from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression

DATASETS_DIRECTORY = "datasets"
CURRENCY_DIRECTORY = "cop"
DATASET_FILENAME = "all_data.csv"
OUTPUT_DIRECTORY = "output2"
TARGET_FEATURE = "USDCOP Curncy"
FORWARD_POINT_FEATURE = "CLN1W BGN Curncy"
TARGET = "Target"
FEATURES = [
    "Date",
    "EURUSD Curncy",
    "GBPUSD Curncy",
    "USDJPY Curncy",
    "USDMXN Curncy",
    "USGG2YR Index",
    "USGG10YR Index",
    "CO1 Comdty",
    "CU1 Comdty",
    "XAU Curncy",
    "CLNI3M Curncy",
    "VIX Index",
    "ES1 Index",
    "NQ1 Index",
    "COLCAP Index",
    "DXY Curncy",
    "COLOM CDS USD SR 5Y D14 Corp",
    "EURUSDV1M BGN Curncy",
    "W 1 COMB Comdty",
    "C 1 COMB Comdty",
    "KC1 COMB Comdty",
    "USGGBE2 Index",
    "USGGBE10 Index",
    "CESIUSD Index",
]

ROLLING_WINDOW = 30

TARGET_RANGE_MIN = -5
TARGET_RANGE_MAX = 5

ROUND_DECIMALS = 5

FORWARD_POINT_DIVIDER = 10000

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
        PLSRegression: {
            "n_components": [5, 10, 20],
            "scale": [True, False],
            "max_iter": [500, 1000, 2000],
        }
    },
    {
        MLPRegressor: {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [0.1, 0.01, 0.001],
            "max_iter": [500, 1000, 2000],
        }
    },
]
