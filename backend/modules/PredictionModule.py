from typing import Dict

import numpy as np


from .ModuleInterface import Module
from .dash import PredictionDashboard

from .ModelManipulator import ModelManipulator


class PredictionModule(Module, PredictionDashboard):
	
	def _prepare_dashboard_settings(self):

		# tmp
        x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(x, np.array([1, 2])) + 3

        self.model = ModelManipulator(model_type=settings['model'], x, y).create()

        super().__init__(settings)
