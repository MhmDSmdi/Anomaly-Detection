from scipy.io import loadmat
data = loadmat("./dataset/arrhythmia.mat")
X = data['X']
y = data['y']
print(len(X[0]))