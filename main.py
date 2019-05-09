import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})
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

    models = [MLPClassifier([100, 100]), MLPClassifier([50, 50])]

    sl = modelSelector(x_train, y_train, x_test, y_test, models, dict)
    sl.eval_models()

    print("main finished")

if __name__ == "__main__":
    main()
