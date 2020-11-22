from typing import Dict

import numpy as np

from .ModuleInterface import Module
from .PredictionDashboardSwitcher import DahsboardSwitcher

from .ModelManipulator import ModelManipulator


ModelDashboard = DahsboardSwitcher().choose()

class PredictionModule(Module, ModelDashboard):

    def _prepare_data(self):
        self.pp.preprocess()
        return self.pp.df

    def _prepare_dashboard_settings(self):

        # tmp
        x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(x, np.array([1, 2])) + 3

        self.model = ModelManipulator(
            x=x, y=y, model_type=settings['model']).create()

    def _prepare_dashboard(self):
        pass
