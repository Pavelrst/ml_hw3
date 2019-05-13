import matplotlib.pyplot as plt
from data_provider import dataProvider
from model_selector import modelSelector

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def main():
    dp = dataProvider()
    dp.test_for_nans()
    # dict = dp.get_vote_dict()
    x_train, y_train = dp.get_train_xy()
    x_val, y_val = dp.get_val_xy()
    x_test, y_test = dp.get_test_xy()
    models = [svm.SVC(kernel='rbf', C=0.8, gamma='auto'),
              KNeighborsClassifier(n_neighbors=85, weights='distance'),
              RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_split=0.01)]
    model_names = ['SVM_rbf',
                   'KNN_distance',
                   'Random_Forest']
    #
    sl = modelSelector(x_train, y_train, x_val, y_val, x_test, y_test, models, model_names)
    sl.fit()
    sl.score_who_win()
    sl.score_vote_prediction()
    sl.score_division_prediction()

if __name__ == "__main__":
    main()
