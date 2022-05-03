import pandas as pd


class loadDataUtils():
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.trainset, self.testset = self.__load(self.train_path, self.test_path)

    def __load(self, train_path, test_path):
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
        return data_train, data_test

    def get_train_and_test(self):
        return self.trainset, self.testset

    def reload_data(self):
        pass

    def get_trainset(self):
        return self.trainset

    def get_testset(self):
        return self.testset

    def set_train_path(self, train_path):
        self.train_path = train_path

    def set_test_path(self, test_path):
        self.test_path = test_path
