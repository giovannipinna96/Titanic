import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class preprocess:
    def __init__(self, data_train, data_test=None):
        self.data_train = data_train
        self.data_test = data_test

        self.sc_Age = StandardScaler()
        self.sc_Fare = StandardScaler()

        self.mean_Miss = None
        self.mean_Mr = None
        self.mean_Mrs = None
        self.mean_Master = None
        self.mean_Dr = None
        self.mean_Rev = None
        self.mean_male = None
        self.mean_female = None
        self.mode_fare = data_train['Fare'].mean()

    def __extract_info_text(self, data):
        data['cabin_multiple'] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
        data['cabin_letter'] = data.Cabin.apply(lambda x: 0 if pd.isna(x) else str(x)[0])
        data['name_title'] = data.Name.apply(lambda x: re.sub('[.]', '', str(re.findall('[a-zA-Z]+[.]*', x)[0])))
        data['name_title'] = data.name_title.apply(
            lambda x: 'Rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev'] else x)

        self.mean_Miss = self.data_train.Age[self.data_train['name_title'] == 'Miss'].mean()
        self.mean_Mr = self.data_train.Age[self.data_train['name_title'] == 'Mr'].mean()
        self.mean_Mrs = self.data_train.Age[self.data_train['name_title'] == 'Mrs'].mean()
        self.mean_Master = self.data_train.Age[self.data_train['name_title'] == 'Master'].mean()
        self.mean_Dr = self.data_train.Age[self.data_train['name_title'] == 'Dr'].mean()
        self.mean_Rev = self.data_train.Age[self.data_train['name_title'] == 'Rev'].mean()
        self.mean_male = self.data_train.Age[self.data_train['Sex'] == 'male'].mean()
        self.mean_female = self.data_train.Age[self.data_train['Sex'] == 'female'].mean()

        return data

    @staticmethod
    def __drop_useless_col(data):
        data.dropna(subset=['Embarked'], inplace=True)
        data.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True)

        return data

    def __fillna_age(self, data):
        data.Age[data['name_title'] == 'Miss'] = data.Age[data['name_title'] == 'Miss'].fillna(self.mean_Miss)
        data.Age[data['name_title'] == 'Mr'] = data.Age[data['name_title'] == 'Mr'].fillna(self.mean_Mr)
        data.Age[data['name_title'] == 'Mrs'] = data.Age[data['name_title'] == 'Mrs'].fillna(self.mean_Mrs)
        data.Age[data['name_title'] == 'Master'] = data.Age[data['name_title'] == 'Master'].fillna(self.mean_Master)
        data.Age[data['name_title'] == 'Dr'] = data.Age[data['name_title'] == 'Dr'].fillna(self.mean_Dr)
        data.Age[data['name_title'] == 'Rev'] = data.Age[data['name_title'] == 'Rev'].fillna(self.mean_Rev)

        data.Age[data['Sex'] == 'male'] = data.Age[data['Sex'] == 'male'].fillna(self.mean_male)
        data.Age[data['Sex'] == 'female'] = data.Age[data['Sex'] == 'female'].fillna(self.mean_female)

        return data

    @staticmethod
    def __create_dummy(data):
        all_dummy = pd.get_dummies(data)
        return all_dummy

    def __process_data(self):
        self.data_train.drop_duplicates()

        self.data_train = self.__extract_info_text(self.data_train)
        self.data_train = self.__drop_useless_col(self.data_train)
        self.data_train = self.__fillna_age(self.data_train)
        self.data_train['Fare'] = self.data_train['Fare'] = np.log(self.data_train.Fare + 1)
        self.data_train['Fare'] = self.sc_Fare.fit_transform(self.data_train[['Fare']])
        self.data_train['Age'] = self.sc_Age.fit_transform(self.data_train[['Age']])
        self.data_train = self.__create_dummy(self.data_train)

        if self.data_test is not None:
            self.data_test = self.__extract_info_text(self.data_test)
            self.data_test = self.__drop_useless_col(self.data_test)
            self.data_test = self.__fillna_age(self.data_test)
            self.data_test['Fare'] = self.data_test['Fare'].fillna(self.mode_fare)
            self.data_test['Fare'] = self.data_test['Fare'] = np.log(self.data_test.Fare + 1)
            self.data_test['Fare'] = self.sc_Fare.transform(self.data_test[['Fare']])
            self.data_test['Age'] = self.sc_Age.transform(self.data_test[['Age']])
            self.data_test = self.__create_dummy(self.data_test)

    def __process_line(self):
        self.data_train = self.__extract_info_text(self.data_train)
        self.data_train = self.__drop_useless_col(self.data_train)
        self.data_train['Fare'] = self.data_train['Fare'] = np.log(self.data_train.Fare + 1)
        self.data_train['Fare'] = self.sc_Fare.transform(self.data_train[['Fare']])
        self.data_train['Age'] = self.sc_Age.transform(self.data_train[['Age']])
        self.data_train = self.__create_dummy(self.data_train)

    def get_data_train(self):
        return self.data_train

    def get_data_test(self):
        return self.data_test

    def get_data(self):
        return self.data_train, self.data_test

    def do_preprocess(self):
        self.__process_data()

    def do_preprocess_for_line(self):
        self.__process_line()

    def set_sc_Age(self, sc_Age):
        self.sc_Age = sc_Age

    def set_sc_Fare(self, sc_Fare):
        self.sc_Fare = sc_Fare

    def get_sc_Fare(self):
        return self.sc_Fare

    def get_sc_Age(self):
        return self.sc_Age