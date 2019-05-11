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
    models = [svm.SVC(kernel='rbf', C=1),
              KNeighborsClassifier(n_neighbors=3, weights='uniform'),
              KNeighborsClassifier(n_neighbors=3, weights='distance'),
              RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=0.1),
              MLPClassifier([24], random_state=1)]
    model_names = ['SVM_rbf',
                   'KNN_uniform',
                   'KNN_distance',
                   'Random_Forest',
                   'MLP[24]']
    #
    sl = modelSelector(x_train, y_train, x_val, y_val, x_test, y_test, models, model_names)
    sl.fit()
    sl.score_who_win()
    sl.score_vote_prediction()
    sl.score_division_prediction()

if __name__ == "__main__":
    main()
