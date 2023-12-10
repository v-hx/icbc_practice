import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale

from config import (
    FEATURES,
    TARGET_FEATURE,
    FORWARD_POINT_FEATURE,
    FORWARD_POINT_DIVIDER,
    TARGET,
    TARGET_RANGE_MIN,
    TARGET_RANGE_MAX,
    ROUND_DECIMALS,
)


class Processor:
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.y_train = None

    # Possible values: pearson, kendall, spearman
    def get_correlated_features(self, method="pearson", threshold=0.95):
        # Create correlation matrix
        corr_matrix = self.data[FEATURES].corr(method=method, numeric_only=True).abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(corr_matrix, k=1).astype(bool))
        # Find features with correlation greater than threshold
        return [column for column in upper.columns if any(upper[column] > threshold)]

    def to_datetime(self, feature="Date"):
        self.data[feature] = pd.to_datetime(self.data[feature])

    def dropna(self):
        self.data.dropna(how="any", inplace=True)

    def ffill(self, columns):
        self.data.loc[:, columns] = self.data.loc[:, columns].ffill()

    def from_float64_to_float32(self):
        float64_cols = list(self.data.select_dtypes(include="float64"))
        self.data[float64_cols] = self.data[float64_cols].astype("float32")

    def create_change_feature(self):
        self.data["Change"] = (
            (
                self.data[TARGET_FEATURE].shift(-1)
                - (
                    self.data[TARGET_FEATURE]
                    + self.data[TARGET_FEATURE]
                    / FORWARD_POINT_DIVIDER
                    / TARGET_RANGE_MAX
                )
            )
            / self.data[TARGET_FEATURE]
        ).round(ROUND_DECIMALS)

    def create_target(self):
        self.data[TARGET] = minmax_scale(
            self.data["Change"], feature_range=(TARGET_RANGE_MIN, TARGET_RANGE_MAX)
        ).round()

    def preprocess(self):
        correlated_features = self.get_correlated_features()
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
        self.ffill([TARGET_FEATURE, FORWARD_POINT_FEATURE])
        self.dropna()
        self.from_float64_to_float32()
        self.create_change_feature()
        self.ffill(["Change"])
        self.create_target()
        self.preprocess()
