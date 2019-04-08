from scipy.io import loadmat

data = loadmat("./arrhythmia.mat")
X = data['X']
y = data['y']


def load_data_set(train_size, test_size):
    X_train = X[: train_size, :]
    X_test = X[train_size:, :]
    y_train = y[: train_size]
    y_test = y[train_size:]
    return (X_train, X_test), (y_train, y_test)


(X_train, X_test), (y_train, y_test) = load_data_set(300, 152)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)