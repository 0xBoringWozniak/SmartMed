from typing import Dict

import pathlib
import sys

import pandas as pd

from sklearn import preprocessing

# logging decorator
from SmartMedApp.logs.logger import debug


class ExtentionFileException(Exception):
    pass


class PandasPreprocessor:
    '''Class to preprocessing any datasets'''

    def __init__(self, settings: Dict):
        self.settings = settings  # settings['data']
        self.__read_file()
        self.numerics_list = {'int16', 'int32', 'int', 'float', 'bool',
                              'int64', 'float16', 'float32', 'float64'}

    @debug
    def __read_file(self):
        ext = pathlib.Path(self.settings['path']).suffix

        if ext == '.csv':
            self.df = pd.read_csv(self.settings['path'])

            if len(self.df.columns) <= 1:
                self.df = pd.read_csv(self.settings['path'], sep=';')

        elif ext == '.xlsx':
            self.df = pd.read_excel(self.settings['path'])

        elif ext == '.tcv':
            self.df = pd.read_excel(self.settings['path'], sep='\t')

        else:
            raise ExtentionFileException

    @debug
    def preprocess(self):
        self.fillna()
        self.encoding()
        self.scale()

    @debug
    def fillna(self):
        value = self.settings['preprocessing']['fillna']
        if value == 'mean':
            for col in self.df.columns:
                if self.df[col].dtype in self.numerics_list:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    self.df[col] = self.df[col].fillna(
                        self.df[col].mode().values[0])
        elif value == 'median':
            for col in self.df.columns:
                if self.df[col].dtype in self.numerics_list:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(
                        self.df[col].mode().values[0])
        elif value == 'droprows':
            self.df = self.df[col].dropna()

    @debug
    def encoding(self):
        method = self.settings['preprocessing']['encoding']
        if method == 'label_encoding':
            transformer = preprocessing.LabelEncoder()

        for column in self.df.select_dtypes(exclude=self.numerics_list):
            transformer.fit(self.df[column].astype(str).values)
            self.df[column] = transformer.transform(
                self.df[column].astype(str).values)

    @debug
    def scale(self):
        method = self.settings['preprocessing']['scaling']
        if method:
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.df)
            self.df = scaler.transform(self.df)
        else:
            pass

    def get_numeric_df(self, df):
        return df.select_dtypes(include=self.numerics_list)

    def get_categorical_df(self, df):
        return df.select_dtypes(exclude=self.numerics_list)
