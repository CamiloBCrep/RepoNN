from keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X, y = zip(*random.sample(list(zip(X_train, y_train)), 2000))

    X, y = np.array(X, dtype='float64'), np.array(y, dtype='float64')
    X = np.reshape(X, (X.shape[0], 28, 28, 1))
    X = MinMaxScaler().fit_transform(X.reshape(-1, 28 * 28)).reshape(-1, 28, 28, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    y_train_value=y_train
    y_train = to_categorical(y_train)
    return X_train, X_test, y_train, y_test, y_train_value