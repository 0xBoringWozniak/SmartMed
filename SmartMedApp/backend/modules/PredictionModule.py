import pandas as pd
import numpy as np

import sklearn.model_selection as sm

from .ModuleInterface import Module
from .dash import PredictionDashboard
from .ModelManipulator import ModelManipulator
from .dataprep import PandasPreprocessor


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

    def _prepare_dashboard_settings(self):
        if self.settings['model'] == 'linreg' or self.settings['model'] == 'logreg':
            names = self.pp.df.columns.tolist()
            names.remove(self.settings['variable'])
            self.df_X = pd.DataFrame()
            for name in names:
                self.df_X = pd.concat([self.df_X, self.pp.df[name]], axis=1)
            self.df_Y = self.pp.df[self.settings['variable']]
            dfX_train, dfX_test, dfY_train, dfY_test = sm.train_test_split(
                self.df_X, self.df_Y, test_size=0.3, random_state=42)
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

        elif self.settings['model'] == 'roc':
            names = self.pp.df.columns.tolist()
            names.remove(self.settings['variable'])
            self.df_X = pd.DataFrame()
            self.df_Y = self.pp.df[self.settings['variable']]
            for name in names:
                self.df_X = pd.concat([self.df_X, self.pp.df[name]], axis=1)
            settings = dict()

            # prepare metrics as names list from str -> bool
            settings['path'] = []
            settings['preprocessing'] = []
            settings['model'] = []
            settings['metrics'] = []
            settings['graphs'] = []
            settings['spec_and_sens'] = []
            settings['spec_and_sens_table'] = []
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
                elif metric == 'auc' or metric == 'diff_graphics' or metric == 'paint':
                    settings['graphs'].append(metric)
                elif metric == 'spec_and_sens':
                    settings['spec_and_sens'] = self.settings['spec_and_sens']
                elif metric == 'spec_and_sens_table':
                    settings['spec_and_sens_table'] = self.settings[
                        'spec_and_sens_table']
                elif self.settings[metric]:
                    settings['metrics'].append(metric)
    #        print('sett', settings)

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
