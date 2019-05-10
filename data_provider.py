import os
import pandas as pd
from sklearn.model_selection import train_test_split
import inspect
import re

class dataProvider():
    def __init__(self, input_path=''):
        self.train_size = None
        self.val_size = None
        self.test_size = None

        if input_path == '':
            delimiter = ''
        else:
            delimiter = '\\'
        self.train_set = pd.read_csv(input_path+delimiter+'train_transformed.csv')
        self.val_set = pd.read_csv(input_path+delimiter+'validation_transformed.csv')
        self.test_set = pd.read_csv(input_path+delimiter+'test_transformed.csv')

        # preparing dataset to model:
        self.y_train = self.train_set.pop('Vote').values
        self.y_val = self.val_set.pop('Vote').values
        self.y_test = self.test_set.pop('Vote').values

        self.x_train = self.train_set.values
        self.x_val = self.val_set.values
        self.x_test = self.test_set.values

        self.test_set_indices = self.test_set.index.values
        self.feature_names = self.train_set.columns


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

    def get_sets_as_pd(self):
        return self.train_set, self.val_set, self.test_set

    def get_train_xy(self):
        return self.x_train, self.y_train

    def get_val_xy(self):
        return self.x_val, self.y_val

    def get_test_xy(self):
        return self.x_test, self.y_test
