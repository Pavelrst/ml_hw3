from graphic_utils import *


def transform_label(_df, label):
    '''
    transform 'Vote' column to integer values
    '''
    _df[label] = _df[label].astype("category").cat.rename_categories(range(_df[label].nunique())).astype(int)


def scale_min_max(data_set, feature, f_min, f_max):
    '''
    scales data set at feature by min max with given min and max
    '''
    assert isinstance(data_set, pd.DataFrame)
    denominator = float(f_max-f_min)
    data_set[feature] = 2 * ((data_set[feature] - f_min) / denominator) - 1


def scale_zscore(data_set, feature, mean, std):
    '''
    scales data set at feature by zscore with given mean and std
    '''
    assert isinstance(data_set, pd.DataFrame)
    data_set[feature] = (data_set[feature] - mean) / float(std)


def scale_sets(train, val, test, gaussian_features, non_gaussian_features):
    '''
    Scales numeric features of all three sets
    some features are scled by min max, others by z score
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    train_and_val = pd.concat([train, val])

    for f in non_gaussian_features:
        f_max = train_and_val[f].max()
        f_min = train_and_val[f].min()
        for data_set in (train, val, test):
            scale_min_max(data_set, f, f_min, f_max)

    for f in gaussian_features:
        f_mean = train_and_val[f].mean()
        f_std = train_and_val[f].std()
        for data_set in (train, val, test):
            scale_zscore(data_set, f, f_mean, f_std)

    return train, val, test


def split_category_to_bits(data_set, cat_feature):
    '''
    splits a categorical feature with N values to N bits
    Currently unused
    '''
    assert isinstance(data_set, pd.DataFrame)
    for cat in data_set[cat_feature].unique():
        data_set["Is_" + cat_feature + "__" + cat] = (data_set[cat_feature] == cat).astype(int)
    del data_set[cat_feature]


def ___transform_categoric(data_set):
    '''
    Transform categorical features to numeric form
    '''
    assert isinstance(data_set, pd.DataFrame)
    assert data_set.isna().sum().sum() == 0
    transform_label(data_set, "Vote")


def transform_categoric(train, val, test):
    '''
    Transforms categoric variables
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    for data_set in (train, val, test):
        ___transform_categoric(data_set)
