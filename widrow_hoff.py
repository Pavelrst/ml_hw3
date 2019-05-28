import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification

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

    def fit(self, x_data, y_data, iterative=False):
        # create a list of weights matrices.
        self.weights_list = self._generate_weights(x_data)
        y_data_list = self._generate_labels(y_data, self.num_classes)

        if iterative:
            for epoch in range(self.epochs):
                # for each matrix, perform gradient descent with LMS loss
                for i in range(self.num_classes):
                    y = y_data_list[i]
                    self._fit_binary_sgd(x_data, y, i)
                # As a result we get num_classes weights matrices.
        else:
            for class_idx in range(self.num_classes):
                y = y_data_list[class_idx]
                # Analytic solution
                # w = XTX^-1XTY
                # Using pseudo inverse
                temp = np.matmul(np.transpose(x_data), x_data)
                try:
                    temp = np.matmul(np.linalg.inv(temp), np.transpose(x_data))
                except:
                    temp = np.matmul(np.linalg.pinv(temp), np.transpose(x_data))
                self.weights_list[class_idx] = np.matmul(temp, y)

    def _fit_binary_sgd(self, x_data, y_data, class_idx):
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


    def _generate_labels(self, y_data, num_classes):
        label_list = []
        for cls in range(num_classes):
            new_y_data = np.zeros_like(y_data)
            for id, val in enumerate(y_data):
                if val == cls:
                    new_y_data[id] = 1
            label_list.append(new_y_data)
        return label_list

def eval_dataset(dataset, num_classes, path, title='title'):
    heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
    rounds = 20
    try:
        X, y = dataset.data, dataset.target
    except:
        X, y = dataset

    classifiers = [
        ("Perceptron (One vs all)", OneVsRestClassifier(Perceptron(tol=1e-3), n_jobs=None)),
        ("LMS (One vs all)", Adaline(num_classes))
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
    plt.title(title)
    fig = plt.gcf()
    fig.savefig(path, bbox_inches='tight')
    plt.show()

def main():
    # eval_dataset(datasets.load_iris(), 3,
    #              path='weirdo_hoff_plots\\iris.png',
    #              title='Iris dataset results')
    # eval_dataset(datasets.load_digits(), 10,
    #              path='weirdo_hoff_plots\\digits.png',
    #              title='Digits dataset results')

    dataset1 = make_classification(n_samples=500,
                                   n_features=2,
                                   n_informative=2,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_classes=2,
                                   n_clusters_per_class=1,
                                   weights=None,
                                   flip_y=0.2,
                                   class_sep=50,
                                   hypercube=False,
                                   shift=0.0,
                                   scale=1.0,
                                   shuffle=True,
                                   random_state=None)
    dataset_name = 'dataset6'

    eval_dataset(dataset1, 2,
                  path='weirdo_hoff_plots\\'+dataset_name+'_results.png',
                  title=dataset_name+' results')

    X, y = dataset1



    first_class_x1 = []
    first_class_x2 = []
    second_class_x1 = []
    second_class_x2 = []
    for sample, label in zip(X, y):
        if label == 0:
            first_class_x1.append(sample[0])
            first_class_x2.append(sample[1])
        else:
            second_class_x1.append(sample[0])
            second_class_x2.append(sample[1])
    plt.scatter(first_class_x1, first_class_x2, label='class A')
    plt.scatter(second_class_x1, second_class_x2, label='class B')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(dataset_name)
    plt.legend()
    fig = plt.gcf()
    fig.savefig('weirdo_hoff_plots\\'+dataset_name+'.png', bbox_inches='tight')
    plt.show()





if __name__ == "__main__":
    main()