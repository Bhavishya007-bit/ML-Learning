"""Transformers learn statistics from training rows only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted

CAT = [f"cat{i}" for i in range(1, 117)]
NUM = [f"cont{i}" for i in range(1, 15)]
FEATURES = CAT + NUM


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Encode category counts / training size; unseen values map to zero.

    Missing values are a distinct category (None). Equal-frequency categories
    collide: this is a documented limitation, not an ordinal encoding.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.n_features_in_ = frame.shape[1]
        self.maps_ = []
        for col in frame:
            values = frame[col].astype(object).where(frame[col].notna(), None)
            self.maps_.append(values.value_counts(normalize=True, dropna=False).to_dict())
        return self

    def transform(self, X):
        check_is_fitted(self, "maps_")
        frame = pd.DataFrame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError("Categorical feature count changed")
        columns = []
        for (_, values), mapping in zip(frame.items(), self.maps_):
            values = values.astype(object).where(values.notna(), None)
            columns.append(values.map(mapping).fillna(0).to_numpy(dtype=float))
        return np.column_stack(columns)


def preprocessor(n_train, seed):
    return ColumnTransformer([
        ("category_frequency", FrequencyEncoder(), CAT),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("quantiles", QuantileTransformer(
                n_quantiles=min(1000, n_train), output_distribution="normal",
                random_state=seed)),
        ]), NUM),
    ], remainder="drop")
