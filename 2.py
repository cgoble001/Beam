import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge


def data_prep(df):
    # Split the data
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_trainX = df_train.iloc[:, 2:]
    df_trainY = df_train.iloc[:, 0]
    df_testX = df_test.iloc[:, 2:]
    df_testY = df_test.iloc[:, 0]
    # Impute missing values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(df_trainX)
    df_trainX = imp_mean.transform(df_trainX)
    imp_mean.fit(df_testX)
    df_testX = imp_mean.transform(df_testX)
    # Normalise the data
    scalar = StandardScaler()
    scalar.fit(df_trainX)
    df_trainX = scalar.transform(df_trainX)
    df_testX = scalar.transform(df_testX)
    return df_trainX, df_trainY, df_testX, df_testY


def dim_red(df_trainX, df_testX):
    # Dimensionality reduction maintaining 95% of the variance using principle component analysis
    pca = PCA(.95)
    pca.fit(df_trainX)
    df_trainX = pca.transform(df_trainX)
    df_testX = pca.transform(df_testX)
    return df_trainX, df_testX


def train(df_trainX, df_trainY):
    # train data
    krr = KernelRidge()
    return krr.fit(df_trainX, df_trainY)


if __name__ == "__main__":
    df = pd.read_csv("crypto_train_research_scientist.csv")
    print('1')
    df_trainX, df_trainY, df_testX, df_testY = data_prep(df)
    print('2')
    krr = KernelRidge()
    krr.fit(df_trainX, df_trainY)
    y_predict = krr.predict(df_testX)
    print('3')
    decision = np.zeros((np.size(y_predict)))
    print('4')
    for i in range(np.size(y_predict)):
        if y_predict[i] > 0.05:
            decision[i] = 1
        elif y_predict[i] < -0.05:
            decision[i] = -1
        else:
            decision[i] = 0
    print(np.dot(decision, y_predict))
    print(sum(decision*df_testY))

