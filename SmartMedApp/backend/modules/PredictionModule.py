import pandas as pd
import numpy as np

from .ModuleInterface import Module
from .dash import PredictionDashboard
from .ModelManipulator import ModelManipulator
from .dataprep import PandasPreprocessor
import sklearn.model_selection as sm

class PredictionModule(Module, PredictionDashboard):

    def _prepare_data(self):
        prep = {'fillna': self.settings['preprocessing'],
                'encoding': 'label_encoding',
                'scaling': False}
        dict_pp = {
            'preprocessing': prep,
            'path': self.settings['path'],
            'fillna': self.settings['preprocessing']
        }
        self.pp = PandasPreprocessor(dict_pp)
        self.pp.preprocess()

        return self.pp.df
        pass

    def _prepare_dashboard_settings(self):
        names = self.pp.df.columns.tolist()
        names.remove(self.settings['variable'])
        self.df_X = pd.DataFrame()
        for name in names:
            self.df_X = pd.concat([self.df_X, self.pp.df[name]], axis=1)
        self.df_Y = self.pp.df[self.settings['variable']]
        dfX_train, dfX_test, dfY_train, dfY_test = sm.train_test_split(self.df_X, self.df_Y, test_size=0.3, random_state=42)
        self.df_X_train = dfX_train
        self.df_X_test = dfX_test
        self.df_Y_train = dfY_train
        self.df_Y_test = dfY_test
        self.model = ModelManipulator(
            x=self.df_X_train, y=self.df_Y_train, model_type=self.settings['model']).create()
        self.model.fit()
        self.mean = sum(dfY_test) / len(dfY_test)

        settings = dict()

        # prepare metrics as names list from str -> bool
        settings['path'] = []
        settings['preprocessing'] = []
        settings['model'] = []
        settings['metrics'] = []
        settings['y'] = []
        settings['x'] = self.pp.df.columns.tolist()

        for metric in self.settings.keys():
            if metric == 'model':
                settings['model'] = self.settings['model']
            elif metric == 'path':
                settings['path'] = self.settings['path']
            elif metric == 'preprocessing':
                settings['preprocessing'] = self.settings['preprocessing']
            elif metric == 'variable':
                settings['y'] = self.settings['variable']
                settings['x'].remove(self.settings['variable'])
            elif self.settings[metric]:
                settings['metrics'].append(metric)

        prep = {'fillna': self.settings['preprocessing'],
                'encoding': 'label_encoding',
                'scaling': False}
        dict_pp = {
            'preprocessing': prep,
            'path': self.settings['path'],
            'fillna': self.settings['preprocessing']
        }
        settings['data'] = dict_pp

        return settings



    def _prepare_dashboard(self):
        pass
