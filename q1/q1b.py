from typing import List, Union
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

import flask

from flask_cors import CORS
import numpy as np
from sklearn.metrics import mean_absolute_error

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')


def create_data_splits(df: pd.DataFrame, test_size_absolute: int):
    """Slices input data into training and test splits. Conventional terminology adopted."""
    df_train, df_test = train_test_split(df, test_size=test_size_absolute)
    return df_train, df_test


def train_model(df_train, target_column_name: str, ignored_columns: List[str]) -> Union[RegressorMixin, ClassifierMixin]:
    """Selects and trains an appropriate supervised learning model with the supplied input data."""
    X = df_train.iloc[:, 2:19]
    print(df_train)
    print(X)
    imp_mean.fit(X)
    y = df_train[target_column_name]
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    X = imp_mean.transform(X)
    krr = KernelRidge(kernel=kernel)
    krr.fit(X, y)
    return krr


def test_model(
    df_test: pd.DataFrame, model: Union[RegressorMixin, ClassifierMixin]
) -> pd.DataFrame:
    """Tests the model on a held out test set"""

    X = df_test.iloc[:, 2:19]
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    y = df_test["R_T1W"]
    X = imp_mean.transform(X)
    score = model.score(X, y)
    return score


if __name__ == "__main__":
    df = pd.read_csv("sample_data_format.csv")
    df_train, df_test = create_data_splits(df=df, test_size_absolute=2000)
    model = train_model(
        df_train, target_column_name="R_T1W", ignored_columns=["Date", "Coin"]
    )

    # save model
    ''' joblib.dump(model, 'model1.pk1')
    model = joblib.load('model1.pk1')'''
    results = test_model(df_test=df_test, model=model)
    print(results)
    # running REST interface for direct test
