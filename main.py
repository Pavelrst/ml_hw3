import matplotlib.pyplot as plt
from data_provider import dataProvider
from sklearn.neural_network import MLPClassifier
from model_selector import modelSelector

def main():
    dp = dataProvider('ElectionsData.csv', '.', 0.7, 0.15, 0.15)
    dp.load_and_split()
    dp.sets_drop_nans_dont_use()
    dp.test_for_nans()
    dict = dp.get_vote_dict()
    x_train, y_train, x_test, y_test, feature_names = dp.get_sets_as_xy_dont_use()
    models = [MLPClassifier([10]), MLPClassifier([20]), MLPClassifier([5, 5]), MLPClassifier([10, 5])]
    model_names = ['MLP[10]', 'MLP[20]', 'MLP[5,5]', 'MLP[10,5]']

    sl = modelSelector(x_train, y_train, x_test, y_test, models, model_names ,dict)
    sl.fit()
    #sl.score_who_win()
    #acc = sl.score_vote_prediction()
    sl.score_division_prediction()
    #print(acc)

if __name__ == "__main__":
    main()
