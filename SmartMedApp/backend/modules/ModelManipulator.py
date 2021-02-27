import numpy as np

from .models import *


class ModelChooseException(Exception):
    pass


class ModelManipulator():

    def __init__(self, model_type: str, x: np.array, y: np.array):

        if model_type == 'linreg':
            self.model = LinearRegressionModel(x, y)

        elif model_type == 'logreg':
            self.model = LogisticRegressionModel(x, y)

        elif model_type == 'polyreg':
            self.model = PolynomialRegressionModel(x, y)

        else:
            raise ModelChooseException

    def create(self):
        return self.model
