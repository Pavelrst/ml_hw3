from graphic_utils import *


def transform_vote(_df, label):
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


def scale_sets(train, val, test):
    '''
    Scales numeric features of all three sets
    some features are scled by min max, others by z score
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    gaussian_features = ['Avg_Residancy_Altitude', 'Avg_education_importance', 'Avg_environmental_importance',
                         'Avg_government_satisfaction', 'Number_of_valued_Kneset_members']
    non_gaussian_features = ['Avg_monthly_expense_on_pets_or_plants', 'Weighted_education_rank',
                             'Yearly_ExpensesK']

    for f in non_gaussian_features:
        f_max = train[f].max()
        f_min = train[f].min()
        for data_set in (train, val, test):
            scale_min_max(data_set, f, f_min, f_max)

    for f in gaussian_features:
        f_mean = train[f].mean()
        f_std = train[f].std()
        for data_set in (train, val, test):
            scale_zscore(data_set, f, f_mean, f_std)

    return train, val, test


def split_category_to_bits(data_set, cat_feature):
    '''
    splits a categorical feature with N values to N bits
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
    split_category_to_bits(data_set, "Most_Important_Issue")
    transform_vote(data_set, "Vote")


def transform_categoric(train, val, test):
    '''
    Transforms categoric variables
    '''
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    for data_set in (train, val, test):
        ___transform_categoric(data_set)
