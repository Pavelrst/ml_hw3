import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 5})
from sklearn.model_selection import train_test_split
import os
from filling_nan_utils import *
from transform_util import *
import re


def main():
    train_set, val_set, test_set = load_and_split('ElectionsData.csv', '.')
    train_set, val_set, test_set = set_clean(train_set, val_set, test_set,
                                             verbose=True, graphic=True)
    assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0
    train_set, val_set, test_set = remove_inconsistency(train_set, val_set, test_set)
    vote_names = train_set['Vote']
    train_set, val_set, test_set = data_transformation(train_set, val_set, test_set, False)

    print("main finished")


def load_and_split(input_path, backup_dir, train=0.7, validation=0.15, test=0.15):
    '''
    This function load .csv file, the original is not modified.
    Split the data to – train (50-75%), validation, (25-15%), test (25-10%)
    For each set – Keep a copy of the raw-data in backup path
    :param input_path: path to data file .csv
    :param backup dir: dir for backup of 3 sets.
    :param train: train ratio of dataset
    :param validation: validation ration of dataset
    :param test: test ratio of dataset
    :return: 3 pandas arrays (datasets): train, validation, test
    '''
    all_data = pd.read_csv(input_path)

    all_data_length = all_data.shape[0]

    train_and_val, test_set = train_test_split(all_data, test_size=test, stratify=all_data[['Vote']])
    train_set, val_set = \
        train_test_split(train_and_val, test_size=validation/(validation+train),
                         stratify=train_and_val[['Vote']])

    train_size = train_set.shape[0]
    val_size = val_set.shape[0]
    test_size = test_set.shape[0]
    assert all_data_length == train_size + val_size + test_size
    assert train_size / all_data_length == train
    assert val_size / all_data_length == validation
    assert test_size / all_data_length == test

    train_set.to_csv(os.path.join(backup_dir, 'train_backup.csv'))
    val_set.to_csv(os.path.join(backup_dir, 'val_backup.csv'))
    test_set.to_csv(os.path.join(backup_dir, 'test_backup.csv'))
    return train_set, val_set, test_set


def set_clean(train_set, val_set, test_set, verbose=True, graphic=False):
    """
    - Fill missing values
    - Smooth noisy data
    - identify\remove outliers.
    - remove inconsistency
    :param train: pandas dataframe train set
    :param val: pandas dataframe val set
    :param test: pandas dataframe test set
    :return: cleaned train, val, test
    """
    init_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])

    delete_vals_out_of_range(train_set, val_set, test_set, verbose=True)
    clipped_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert clipped_num_nans > init_num_nans

    train_set, val_set, test_set, redundant_numeric_features, useful_numeric_features = \
        fill_nans_by_lin_regress(train_set, val_set, test_set)
    first_fill_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert clipped_num_nans > first_fill_num_nans

    delete_outliers(train_set, val_set, test_set, useful_numeric_features)
    no_outliers_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert no_outliers_num_nans > first_fill_num_nans

    train_set, val_set, test_set, redundant_numeric_features, useful_numeric_features = \
        fill_nans_by_lin_regress(train_set, val_set, test_set)
    sec_lin_reg_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert sec_lin_reg_num_nans < no_outliers_num_nans

    delete_vals_out_of_range(train_set, val_set, test_set, verbose=True)
    reclipped_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert reclipped_num_nans > sec_lin_reg_num_nans

    pre_drop_num_cols = len(train_set.columns)
    all_sets = [train_set, val_set, test_set]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = data_set.drop(redundant_numeric_features, axis=1)
    [train_set, val_set, test_set] = all_sets
    post_drop_num_cols = len(train_set.columns)
    assert post_drop_num_cols < pre_drop_num_cols

    fill_missing_vals_by_mean(train_set, val_set, test_set, useful_numeric_features)
    num_features_full_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert num_features_full_num_nans < sec_lin_reg_num_nans
    assert sum([s[useful_numeric_features].isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0

    delete_rare_categorical_vals(train_set, val_set, test_set)  # Does nothing on our data set
    fill_categorical_missing_vals(train_set, val_set, test_set)
    assert sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)]) == 0

    return train_set, val_set, test_set


def smooth_noisy_data(train_set, val_set, test_set, verbose=True, graphic=False):
    '''
    This function handles negative values of
    parameters which can't be negative.
    '''
    if graphic:
        show_set_hist(train_set, title='train_set noisy data')
        show_set_hist(val_set, title='val_set noisy data')
        show_set_hist(test_set, title='test_set noisy data')

    init_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    delete_vals_out_of_range(train_set, val_set, test_set)
    clip_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert clip_num_nans > init_num_nans
    delete_outliers(train_set, val_set, test_set)
    outlier_num_nans = sum([s.isna().sum().sum() for s in (train_set, val_set, test_set)])
    assert outlier_num_nans >= clip_num_nans

    if graphic:
        show_set_hist(train_set, title='train_set noisy data')
        show_set_hist(val_set, title='val_set noisy data')
        show_set_hist(test_set, title='test_set noisy data')

    return train_set, val_set, test_set


def remove_inconsistency(train_set, val_set, test_set):
    '''
    Removes columns which have exact same features besides Vote, yet differ on the Vote
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    columns = list(train_set)
    columns.remove('Vote')
    for data_set in (train_set, val_set, test_set):
        data_set.drop_duplicates(columns)
    return train_set, val_set, test_set


def data_transformation(train_set, val_set, test_set, graphic=False):
    """
    - Scaling
    - Normalization (Z-score or min-max)
    - Conversion
    :param train_set:
    :param val_set:
    :param test_set:
    :param how:
    :return:
    """
    if graphic:
        show_set_hist(train_set, title='train_set histogram before scaling')
    train_set, val_set, test_set = scale_sets(train_set, val_set, test_set)
    if graphic:
        show_set_hist(train_set, title='train_set histogram after scaling')
    transform_categoric(train_set, val_set, test_set)
    return train_set, val_set, test_set


def set_reduction(train_set, val_set, test_set):
    '''
    - implement Feature selection:
    - One filter method
    - One wrapper method
    :param train_set:
    :param val_set:
    :param test_set:
    :param how:
    :return: reduced set
    '''
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    selected_features = select_features(train_set)
    selected_features.append('Vote')
    all_sets = [train_set, val_set, test_set]
    for index, data_set in enumerate(all_sets):
        all_sets[index] = all_sets[index][selected_features]
    [train_set, val_set, test_set] = all_sets
    return train_set, val_set, test_set


def save_datasets(train_set, val_set, test_set):
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(val_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    train_set.to_csv('train_transformed.csv')
    val_set.to_csv('validation_transformed.csv')
    test_set.to_csv('test_transformed.csv')


def export_features_to_csv(features):
    used_features = set()
    for f in features:
        if f == 'Vote':
            continue
        match = re.search('^Is_(.+)__.+$', f)
        if match is not None:
            used_features.add(match.group(1))
        else:
            used_features.add(f)
    used_features = list(used_features)
    used_features.sort()
    txt = ','.join(used_features)
    with open('SelectedFeatures.csv', 'w') as f:
        f.write(txt)


if __name__ == "__main__":
    main()
