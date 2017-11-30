from sklearn.neural_network import MLPClassifier
from new_commons import *

name = "DNN_sklearn"


MLPC = MLPClassifier(hidden_layer_sizes=(100,100), solver='sgd', max_iter=30, early_stopping=False, learning_rate_init=0.01)

X_train, y_train = prepareTrainData()
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / (std + 0.001)
evaluateClassifier(MLPC, X_train, y_train, name)
printSubmission(MLPC, X_train, y_train, name, mean=mean, std=std)
