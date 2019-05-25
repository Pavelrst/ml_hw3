import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

LR = 0.0001

class Adaline:
    '''
    ADALINE are similar to the perceptron,
    but their transfer function is linear rather than
    hard-limiting. This allows their outputs to take on any value,
    whereas the perceptron output is limited to either 0 or 1.
    perceptron, can only solve linearly separable problems.
    '''
    def __init__(self, num_classes, epochs=1000):
        self.num_classes = num_classes
        self.weights_list = None
        self.epochs = epochs

    def fit(self, x_data, y_data):
        # create a list of weights matrices.
        self.weights_list = self._generate_weights(x_data)
        y_data_list = self._generate_labels(y_data)

        for epoch in range(self.epochs):
            # for each matrix, perform gradient descent with LMS loss
            for i in range(self.num_classes):
                y = y_data_list[i]
                self._fit_binary(x_data, y, i)
            # As a result we get num_classes weights matrices.

    def _fit_binary(self, x_data, y_data, class_idx):
        # Fits one weight, one class
        # SGD on whole x data
        for sample_idx in range(x_data.shape[0]):
            curr_sample = x_data[sample_idx]
            curr_label = y_data[sample_idx]
            wTx = np.matmul(np.array(self.weights_list[class_idx]), curr_sample)
            step = np.multiply((curr_label-wTx), curr_sample)
            self.weights_list[class_idx] = self.weights_list[class_idx] + LR*step

    def predict(self, x_data):
        # Each sample multiplied by each matrix.
        # The label is given according to smallest LMS?
        predicitons = []
        for sample_idx in range(x_data.shape[0]):
            curr_sample = x_data[sample_idx]
            predicitons.append(self._predict_sample(curr_sample))
        return np.array(predicitons)

    def _predict_sample(self, sample):
        # Each model predicts current sample
        highest = -np.inf
        prediction = None
        for class_idx in range(self.num_classes):
            wTx = np.matmul(np.array(self.weights_list[class_idx]), sample)
            if wTx > 0 and wTx > highest:
                highest = wTx
                prediction = class_idx
        return prediction


    def _generate_weights(self, x_data):
        weights = []
        for i in range(self.num_classes):
            w = np.random.rand(x_data.shape[1])
            weights.append(w)
        return weights


    def _generate_labels(self, y_data):
        label_list = []
        for cls in set(y_data):
            new_y_data = np.zeros_like(y_data)
            for id, val in enumerate(y_data):
                if val == cls:
                    new_y_data[id] = 1
            label_list.append(new_y_data)
        return label_list


def main():



    heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
    rounds = 20
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ada = Adaline(len(set(y)))
    ada.fit(X_train, y_train)
    print("acc = ", np.mean(ada.predict(X_test) == y_test))

    classifiers = [
        ("Perceptron", Perceptron(tol=1e-3)),
        ("One vs all", OneVsRestClassifier(Perceptron(tol=1e-3), n_jobs=None)),
        ("LMS", Adaline(len(set(y))))
    ]

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)

    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.show()

if __name__ == "__main__":
    main()