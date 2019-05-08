import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})
from data_provider import dataProvider


def main():
    dp = dataProvider('ElectionsData.csv', '.', 0.7, 0.15, 0.15)
    dp.load_and_split()
    dp.sets_drop_nans_dont_use()
    dp.test_for_nans()
    x_train, y_train, x_test, y_test, feature_names = dp.get_sets_as_xy_dont_use()
    print("main finished")

if __name__ == "__main__":
    main()
