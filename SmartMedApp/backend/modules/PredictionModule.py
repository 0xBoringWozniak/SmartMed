import pandas as pd
import numpy as np

from .ModuleInterface import Module
from .dash import PredictionDashboard
from .ModelManipulator import ModelManipulator


class PredictionModule(Module, PredictionDashboard):

    def _prepare_data(self):
        pass

    def _prepare_dashboard_settings(self):

        # tmp
        x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(x, np.array([1, 2])) + 3

        self.model = ModelManipulator(
            x=x, y=y, model_type=settings['model']).create()

        self.model.fit()

    def _prepare_dashboard(self):
        pass
