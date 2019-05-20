import matplotlib.pyplot as plt
from data_provider import dataProvider
from model_selector import modelSelector
from feature_manipulator import featureManipulator
from cross_validation import crossValidator
import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def main():
    dp = dataProvider()
    dict = dp.get_vote_dict()
    dp.test_for_nans()
    id_train, x_train, y_train = dp.get_train_xy()
    id_val, x_val, y_val = dp.get_val_xy()
    id_test, x_test, y_test = dp.get_test_xy()

    assert set(id_train).intersection(set(id_val)) == set()
    assert set(id_val).intersection(set(id_test)) == set()
    assert set(id_test).intersection(set(id_train)) == set()

    # Cross validation
    # cv = crossValidator(train_x=x_train, train_y=y_train, num_of_folds=3)
    # cv.tuneSVM([10**-5,10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4,10**5], type='coarse')
    # cv.tuneSVM(np.arange(0.1, 100, 0.5), type='fine')
    # cv.tuneKNN(101)
    # cv.tuneNForest(100)
    # cv.tuneDepthForest(50)
    # cv.tuneSplitForest()
    # cv.tuneMLP()

    models = [svm.SVC(kernel='rbf', C=20, gamma='auto', probability=True),
              MLPClassifier([50], activation='relu', max_iter=1000),
              KNeighborsClassifier(n_neighbors=3, weights='distance'),
              RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_split=0.01)]
    model_names = ['SVM_rbf',
                    'MLP[50]',
                    'KNN_distance',
                    'Random_Forest']

    sl = modelSelector(id_train, x_train, y_train,
                       id_val, x_val, y_val,
                       id_test, x_test, y_test,
                       models, model_names, dict)
    sl.fit()
    sl.score_accuracy()
    sl.save_votes_to_csv()
    sl.score_who_win(graphic=True)
    sl.score_transportation_prediction(graphic=True)
    sl.score_division_prediction(graphic=True)

    # One for each
    # sl.predict_winner(x_test)
    # sl.predict_vote_division(x_test)
    # sl.predict_transportation(x_test)
    # sl.draw_conf_matrix()
    # sl.get_test_error()

    # One for all
    sl.score_one_for_all()
    sl.predict_winner(x_test, True)
    sl.predict_vote_division(x_test, True)
    sl.predict_transportation(x_test, True)
    sl.draw_conf_matrix(True)
    sl.get_test_error(True)

    #
    # model = sl.get_best_winner_prediction_model()
    # feature_names = dp.get_feature_names()
    # fml = featureManipulator(model, x_test, y_test, feature_names, party_dict=dict)
    # fml.find_continuous_dramatic_feature()
    # fml.find_binary_dramatic_feature()

if __name__ == "__main__":
    main()
