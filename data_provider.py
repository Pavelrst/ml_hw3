import os
import pandas as pd
from sklearn.model_selection import train_test_split
import inspect
import re

class dataProvider():
    def __init__(self, input_path, backup_dir, train=0.7, validation=0.15, test=0.15):
        self.input_path = input_path
        self.backup_dir = backup_dir
        self.train_ratio = train
        self.validation_ratio = validation
        self.test_ratio = test
        self.all_data_length = None

        self.train_size = None
        self.val_size = None
        self.test_size = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.vote_categories = None
        self.vote_numbers = None
        self.vote_dictionary = None

    def test_for_nans(self):
        assert sum([s.isna().sum().sum() for s in (self.train_set, self.val_set, self.test_set)]) == 0

    def get_vote_dict(self):
        '''
        :return: dictionary which maps 'Vote' category to numbers.
        '''
        if self.vote_dictionary is not None:
            return self.vote_dictionary

        if self.vote_categories is not None:
            categories_list = []
            for cat in self.vote_categories:
                if cat not in categories_list:
                    categories_list.append(cat)
        else:
            print("Warning from", inspect.stack()[0][3], ": self.vote_categories are None!")
            return None

        if self.vote_numbers is not None:
            numbers_list = []
            for num in self.vote_numbers:
                if num not in numbers_list:
                    numbers_list.append(num)
        else:
            print("Warning from", inspect.stack()[0][3], ": self.vote_numbers are None!")
            return None

        self.vote_dictionary = dict(zip(numbers_list, categories_list))
        return self.vote_dictionary

    def sets_drop_nans_dont_use(self):
        '''
        This function simply deletes all NANs
        and converts 'Vote' to numbers
        Don't use this function
        in your final assignment.
        '''
        if self.train_set is not None \
            and self.val_set is not None \
            and self.test_set is not None:

            # for dictionary preparation
            self.vote_categories = self.train_set['Vote']

            self.train_set = self.train_set.dropna()
            self.val_set = self.val_set.dropna()
            self.test_set = self.test_set.dropna()
            pd.options.mode.chained_assignment = None  # default='warn'
            self.train_set['Vote'] = self.train_set['Vote'].astype('category')
            self.train_set['Vote'] = self.train_set['Vote'].cat.codes
            self.val_set['Vote'] = self.val_set['Vote'].astype('category')
            self.val_set['Vote'] = self.val_set['Vote'].cat.codes
            self.test_set['Vote'] = self.test_set['Vote'].astype('category')
            self.test_set['Vote'] = self.test_set['Vote'].cat.codes

            # for dictionary preparation
            self.vote_numbers = self.train_set['Vote']
        else:
            print("Warning from", inspect.stack()[0][3], ": sets are None!")

    def get_sets_as_pd(self):
        return self.train_set, self.val_set, self.test_set

    def get_sets_as_xy_dont_use(self):
        '''
        This function does one hot encoding,
        and label encoding
        :return: X_train, Y_train, X_test, Y_test numpy arrays fuature_names - list
        '''
        # preparing dataset to model:
        y_train = self.train_set.pop('Vote').values
        y_test = self.val_set.pop('Vote').values

        # Temp solution to convert categorical data to numerical
        for col_name in self.train_set.columns:
            if self.train_set[col_name].dtype == 'object':
                self.train_set[col_name] = self.train_set[col_name].astype('category')
                self.train_set[col_name] = self.train_set[col_name].cat.codes
        for col_name in self.val_set.columns:
            if self.val_set[col_name].dtype == 'object':
                self.val_set[col_name] = self.val_set[col_name].astype('category')
                self.val_set[col_name] = self.val_set[col_name].cat.codes

        x_train = self.train_set.values
        x_test = self.val_set.values
        feature_names = self.train_set.columns
        return x_train, y_train, x_test, y_test, feature_names

    def load_and_split(self):
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
        all_data = pd.read_csv(self.input_path)
        self.all_data_length = all_data.shape[0]

        train_and_val, self.test_set = train_test_split(all_data,
                                                   test_size=self.test_ratio,
                                                   stratify=all_data[['Vote']])
        test_size = self.validation_ratio / (self.validation_ratio + self.train_ratio)
        self.train_set, self.val_set = train_test_split(train_and_val,
                                                test_size=test_size,
                                                stratify=train_and_val[['Vote']])

        self.train_size = self.train_set.shape[0]
        self.val_size = self.val_set.shape[0]
        self.test_size = self.test_set.shape[0]

        assert self.all_data_length == self.train_size + self.val_size + self.test_size
        assert self.train_size / self.all_data_length == self.train_ratio
        assert self.val_size / self.all_data_length == self.validation_ratio
        assert self.test_size / self.all_data_length == self.test_ratio

        self.train_set.to_csv(os.path.join(self.backup_dir, 'train_backup.csv'))
        self.val_set.to_csv(os.path.join(self.backup_dir, 'val_backup.csv'))
        self.test_set.to_csv(os.path.join(self.backup_dir, 'test_backup.csv'))

