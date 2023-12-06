import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale

from config import FEATURES, TARGET_FEATURE, TARGET, TARGET_RANGE_MIN, TARGET_RANGE_MAX

pd.options.mode.chained_assignment = None


class Processor:
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.y_train = None

    # pearson, kendall, spearman = get_correlated_features('pearson')
    def get_correlated_features(self, method="pearson", threshold=0.95):
        # Create correlation matrix
        corr_matrix = self.data[FEATURES].corr(method=method, numeric_only=True).abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(corr_matrix, k=1).astype(bool))
        # Find features with correlation greater than threshold
        return [column for column in upper.columns if any(upper[column] > threshold)]

    # Date object -> datetime format
    def to_datetime(self, feature="Date"):
        self.data[feature] = pd.to_datetime(self.data[feature])

    # 1. Keep rows with present target
    def get_with_targer(self):
        self.data = self.data[self.data[TARGET_FEATURE].notna()]

    def fill_with_previous(self):
        # 2. Fill empty values from previous rows
        self.data.ffill(inplace=True)
        # 3. Remove rows with NaN (meanin that there were no previous values for some rows)
        self.data.dropna(how="any", inplace=True)

    # 'BCN1W BGN Curncy'
    def drop_features(self, features):
        # Drop BCN1W BGN Curncy
        self.data.drop(features, axis=1, inplace=True)

    # float64 dtype -> float32
    def from_float64_to_float32(self):
        float64_cols = list(self.data.select_dtypes(include="float64"))
        self.data[float64_cols] = self.data[float64_cols].astype("float32")

    # Change is always NaN for the first row, removing
    # data = data.iloc[1:]
    def create_change_feature(self):
        self.data["Change"] = (
            self.data[TARGET_FEATURE].shift() - self.data[TARGET_FEATURE]
        ) / self.data[TARGET_FEATURE]
        # Change is always NaN for the first row, removing
        self.data = self.data.iloc[1:]

    def create_target(self):
        # Creating target and normalize it
        self.data[TARGET] = minmax_scale(
            self.data["Change"], feature_range=(TARGET_RANGE_MIN, TARGET_RANGE_MAX)
        ).round()

    def preprocess(self):
        correlated_features = self.get_correlated_features()
        self.drop_features(correlated_features)
        features = [
            feature for feature in FEATURES if feature not in correlated_features
        ]
        X = self.data[features] 
        y = self.data[TARGET]
        numerical_features = X.select_dtypes(include=["float32"]).columns
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[("num", numerical_transformer, numerical_features)]
        )

        preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        self.X_train = preprocessing_pipeline.fit_transform(X)
        self.y_train = y

    def do_process(self):
        self.to_datetime()
        self.get_with_targer()
        self.fill_with_previous()
        self.from_float64_to_float32()
        self.create_change_feature()
        self.create_target()
        self.preprocess()
